from typing import Callable, Optional


class ButtonController:
    """
    녹음 버튼과 읽어주기 버튼을 관리한다.

    Raspberry Pi:
        gpiozero를 사용하여 실제 GPIO 버튼을 읽는다.

    MacBook 테스트:
        simulate=True로 실행하여 키보드 입력으로 버튼을 시뮬레이션한다.
    """

    def __init__(
        self,
        record_callback: Callable[[], None],
        tts_callback: Callable[[], None],
        record_pin: int = 17,
        tts_pin: int = 27,
        simulate: bool = False,
    ) -> None:

        self.record_callback = record_callback
        self.tts_callback = tts_callback

        self.record_pin = record_pin
        self.tts_pin = tts_pin

        self.simulate = simulate

        self.record_button = None
        self.tts_button = None

        if not self.simulate:
            self._setup_gpio()

    def _setup_gpio(self) -> None:
        """
        Raspberry Pi GPIO 버튼을 설정한다.
        """

        try:
            from gpiozero import Button
        except ImportError as error:
            raise RuntimeError(
                "gpiozero가 설치되어 있지 않습니다."
            ) from error

        self.record_button = Button(
            self.record_pin,
            pull_up=True,
        )

        self.tts_button = Button(
            self.tts_pin,
            pull_up=True,
        )

        self.record_button.when_pressed = (
            self._on_record_pressed
        )

        self.tts_button.when_pressed = (
            self._on_tts_pressed
        )

    def _on_record_pressed(self) -> None:
        print("[BUTTON] 녹음 버튼")
        self.record_callback()

    def _on_tts_pressed(self) -> None:
        print("[BUTTON] 읽어주기 버튼")
        self.tts_callback()

    def simulate_record_button(self) -> None:
        """
        MacBook 테스트용 녹음 버튼 입력.
        """
        self._on_record_pressed()

    def simulate_tts_button(self) -> None:
        """
        MacBook 테스트용 읽어주기 버튼 입력.
        """
        self._on_tts_pressed()