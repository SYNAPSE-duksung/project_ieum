const audioInput = document.getElementById("audioInput");

const filename = document.getElementById("filename");

const sendBtn = document.getElementById("sendBtn");

const status = document.getElementById("status");

let selectedFile = null;



// --------------------------------------
// 파일 선택
// --------------------------------------

audioInput.addEventListener("change", () => {

    if (audioInput.files.length === 0) {

        selectedFile = null;

        filename.textContent = "선택된 파일 없음";

        return;

    }

    selectedFile = audioInput.files[0];

    filename.textContent = selectedFile.name;

});



// --------------------------------------
// 자막 시작
// --------------------------------------

sendBtn.addEventListener("click", async () => {

    if (!selectedFile) {

        alert("wav 파일을 선택해주세요.");

        return;

    }

    status.textContent = "업로드 중...";

    const buffer = await selectedFile.arrayBuffer();

    chrome.runtime.sendMessage(

        {

            type: "UPLOAD_AUDIO",

            filename: selectedFile.name,

            buffer: Array.from(new Uint8Array(buffer))

        },

        (response) => {

            if (chrome.runtime.lastError) {

                console.error(chrome.runtime.lastError);

                status.textContent = "실패";

                return;

            }

            if (!response) {

                status.textContent = "실패";

                return;

            }

            if (response.success) {

                status.textContent = "완료";

            }

            else {

                status.textContent = "실패";

            }

        }

    );

});