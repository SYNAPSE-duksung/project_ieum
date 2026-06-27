// =========================================
// IEUM Extension - content.js
// =========================================

console.log("IEUM Content Loaded");

// ------------------------------------------
// 자막 박스 생성
// ------------------------------------------

let captionBox = document.getElementById("ieum-caption");

if (!captionBox) {

    captionBox = document.createElement("div");

    captionBox.id = "ieum-caption";

    document.body.appendChild(captionBox);

}

let hideTimer = null;


// ------------------------------------------
// 자막 표시
// ------------------------------------------

function showCaption(text) {

    clearTimeout(hideTimer);

    captionBox.textContent = text;

    captionBox.classList.add("show");

    hideTimer = setTimeout(() => {

        captionBox.classList.remove("show");

    }, 3000);

}


// ------------------------------------------
// 상태 표시
// ------------------------------------------

function showStatus(text) {

    clearTimeout(hideTimer);

    captionBox.textContent = text;

    captionBox.classList.add("show");

}


// ------------------------------------------
// background 메시지 수신
// ------------------------------------------

chrome.runtime.onMessage.addListener((message) => {

    console.log(message);

    switch (message.type) {

        case "SHOW_STATUS":

            showStatus(message.text);

            break;

        case "SHOW_CAPTION":

            showCaption(message.text);

            break;

    }

});