"""Stage one recording ahead without changing source files or VAD."""
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import os
import shutil
import time
import uuid


class LocalPrefetch:
    def __init__(self, rows, source, destination, runner):
        self.rows = rows
        self.source = source.resolve()
        self.destination = destination.resolve()
        self.runner = runner
        if self.source.is_relative_to(self.destination) or self.destination.is_relative_to(self.source):
            raise ValueError('Staging must be separate from sources')
        self.pool = ThreadPoolExecutor(max_workers=1)
        self.future = self.pool.submit(self.stage, rows[0]) if rows else None

    def stage(self, row):
        # Drive streaming can report EINVAL while hydrating an otherwise valid
        # file. Retry with a fresh partial, retaining every failed copy.
        started = time.perf_counter()
        for attempt in range(3):
            try:
                paths, _ = self._stage_once(row)
                return paths, time.perf_counter() - started
            except OSError as exc:
                if exc.errno != 22 or attempt == 2:
                    raise
                delay = 2 ** (attempt + 1)
                print(f'Prefetch read error: {row["recording_id"]}; '
                      f'retry {attempt + 1}/2 in {delay}s; partial preserved: {exc}',
                      flush=True)
                time.sleep(delay)

    def _stage_once(self, row):
        started = time.perf_counter()
        result = {}
        for kind, source in self.runner.paths_for(row, self.source).items():
            self.runner.download(row, kind, source)
            target = self.destination / source.relative_to(self.source)
            target.parent.mkdir(parents=True, exist_ok=True)
            if target.exists():
                self.runner.download(row, kind, target)
            else:
                expected = int(row[kind + '_size_bytes'])
                if shutil.disk_usage(target.parent).free < expected + self.runner.RESERVE:
                    raise ValueError('Insufficient staging space including reserve')
                partial = target.with_name(target.name + '.partial_' + uuid.uuid4().hex)
                print(f'Prefetch copying: {source} -> {partial}; expected={expected} bytes', flush=True)
                with source.open('rb') as src, partial.open('xb') as dst:
                    written = 0
                    last_progress = time.perf_counter()
                    while True:
                        chunk = src.read(8 * 1024 * 1024)
                        if not chunk:
                            break
                        written += len(chunk)
                        if written > expected or shutil.disk_usage(target.parent).free < len(chunk) + self.runner.RESERVE:
                            raise ValueError('Staging byte/space limit exceeded; partial preserved')
                        dst.write(chunk)
                        if time.perf_counter() - last_progress >= 10:
                            print(f'Prefetch progress: {source.name}; {written}/{expected} bytes', flush=True)
                            last_progress = time.perf_counter()
                    dst.flush()
                    os.fsync(dst.fileno())
                if written != expected or source.stat().st_size != expected:
                    raise ValueError('Source/staged size differs; partial preserved')
                # On Windows rename refuses an existing destination.
                if target.exists():
                    raise ValueError('Staging target appeared; partial preserved')
                os.rename(partial, target)
                self.runner.download(row, kind, target)
            result[kind] = target
        return result, time.perf_counter() - started

    def take(self, index):
        started = time.perf_counter()
        paths, stage_seconds = self.future.result()
        wait_seconds = time.perf_counter() - started
        self.future = (self.pool.submit(self.stage, self.rows[index + 1])
                       if index + 1 < len(self.rows) else None)
        return paths, stage_seconds, wait_seconds

    def close(self):
        self.pool.shutdown(wait=True, cancel_futures=True)
