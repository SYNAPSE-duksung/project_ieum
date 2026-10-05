// Replay in Codex functions.exec tool runtime. Metadata-only connector calls.
// Writes a NEW snapshot; apply_patch must not be invoked on an existing snapshot.
// No WAV/JSON contents, downloads, or preprocessing are requested.
const roots = [
  ['TS01','1cf8fINhFjUM-USWK3ZWpbeilrLCQhatO'],
  ['TL01','109kzJDEIK0cPl7xtnCD41t9596q0qsRo'],
  ['TS02','1EUVD3M1NvknCeZQKF3QHMIwf_a2Zl6D_'],
  ['TL02','1fGunxCUzD47j3_jiG74x_hJFIe0fi0Wm'],
  ['TS03','1AMJVoH9I29SAdxyGfdfAWmiY4c6pt6h3'],
  ['TL03','1f5e2zkDySNRMNN9tFwjEOKSD_9jja7c4'],
  ['VS01','1FR4GcYVHcpLS603Pw-mU4UbnAGBfL3vc'],
  ['VL01','1tNLFDovdkIBq1eQ2Bk7srTUqFyT2SJdj']
];
const folders = [];
let queue = roots.map(([dataset,id])=>({dataset,id,path:dataset}));
while (queue.length) {
  const batch = queue.splice(0,16);
  const results = await Promise.allSettled(batch.map(async entry=>{
    const response = await tools.mcp__codex_apps__google_drive_list_folder({
      url:'https://drive.google.com/drive/folders/'+entry.id,top_k:1000
    });
    if(response.isError) throw new Error('Drive listing failed: '+entry.path);
    const files = response.structuredContent.files;
    if(files.length>=1000) throw new Error('Potentially truncated listing: '+entry.path);
    return {...entry,requested_limit:1000,error:false,files:files.map(f=>({
      id:f.id,name:f.title,size:f.size,mime_type:f.mime_type,kind:f.file_or_folder,url:f.url
    }))};
  }));
  for (const result of results) {
    if(result.status!=='fulfilled') throw result.reason;
    folders.push(result.value);
    for(const f of result.value.files.filter(f=>f.kind==='folder')) {
      queue.push({dataset:result.value.dataset,id:f.id,path:result.value.path+'/'+f.name});
    }
  }
  notify({listed_folders:folders.length,remaining_folders:queue.length});
}
const snapshot = {root_id:'19yxQbRr6XQm_j4FDbTh1JkFsNH6FcA9g',
  observed_at:new Date().toISOString(),method:'Google Drive list_folder; metadata only; recursive top_k=1000',folders};
// Choose a new staging filename for each refresh; do not overwrite prior evidence.
text(await tools.apply_patch('*** Begin Patch\n*** Add File: _full_expansion/drive_listing_snapshot_REFRESH.json\n+'
  +JSON.stringify(snapshot)+'\n*** End Patch'));
