// =========================================
// IEUM Extension - background.js
// =========================================

chrome.runtime.onMessage.addListener((message, sender, sendResponse) => {

    if (message.type === "UPLOAD_AUDIO") {

        handleUpload(message)
            .then(() => {

                sendResponse({
                    success: true
                });

            })
            .catch((error) => {

                console.error(error);

                sendResponse({
                    success: false,
                    error: error.toString()
                });

            });

        return true;
    }

});



// ------------------------------------------
// Google Meet 탭 찾기
// ------------------------------------------

async function getMeetTab() {

    const tabs = await chrome.tabs.query({});

    return tabs.find(tab =>

        tab.url &&
        tab.url.startsWith("https://meet.google.com/")

    );

}



// ------------------------------------------
// content.js 로 메시지 보내기
// ------------------------------------------

async function sendToMeet(type, text) {

    const meetTab = await getMeetTab();

    if (!meetTab) {

        console.log("Google Meet 탭을 찾을 수 없습니다.");

        return;

    }

    chrome.tabs.sendMessage(

        meetTab.id,

        {
            type: type,
            text: text
        },

        () => {

            if (chrome.runtime.lastError) {

                console.log(chrome.runtime.lastError.message);

            }

        }

    );

}



// ------------------------------------------
// FastAPI 호출
// ------------------------------------------

async function handleUpload(message) {

    // 상태 표시
    await sendToMeet(

        "SHOW_STATUS",

        "음성 업로드 중..."

    );



    // Array -> Blob

    const uint8 = new Uint8Array(message.buffer);

    const blob = new Blob(

        [uint8],

        {

            type: "audio/wav"

        }

    );



    const formData = new FormData();

    formData.append(

        "audio",

        blob,

        message.filename

    );



    // Whisper

    await sendToMeet(

        "SHOW_STATUS",

        "Whisper 음성 인식 중..."

    );



    const response = await fetch(

        "http://127.0.0.1:8000/speech",

        {

            method: "POST",

            body: formData

        }

    );



    if (!response.ok) {

        throw new Error("FastAPI 호출 실패");

    }



    const result = await response.json();



    // Gemini

    await sendToMeet(

        "SHOW_STATUS",

        "✨ 문장 보정 중..."

    );



    await new Promise(resolve => setTimeout(resolve, 700));



    // 최종 자막

    await sendToMeet(

        "SHOW_CAPTION",

        result.corrected

    );



    console.log(result);

}