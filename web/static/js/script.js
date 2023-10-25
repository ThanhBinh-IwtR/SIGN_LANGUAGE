const video = document.getElementById('video');
const recordButton = document.getElementById('record');
const stopRecordButton = document.getElementById('stopRecord');
const canvas = document.getElementById('canvas');
const streamUrl = '/video_feed';

let stream;
let mediaRecorder;
let chunks = [];

async function startStream() {
    stream = await navigator.mediaDevices.getUserMedia({ video: true });
    video.srcObject = stream;
    recordButton.disabled = false;
}

function startRecording() {
    mediaRecorder = new MediaRecorder(stream);
    mediaRecorder.ondataavailable = (event) => {
        if (event.data.size > 0) {
            chunks.push(event.data);
        }
    };
    mediaRecorder.start();
    recordButton.disabled = true;
    stopRecordButton.disabled = false;
}

function stopRecording() {
    mediaRecorder.stop();
    recordButton.disabled = false;
    stopRecordButton.disabled = true;
    const blob = new Blob(chunks, { 'type' : 'video/webm' });
    chunks = [];
    const url = window.URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = 'recorded.webm';
    document.body.appendChild(a);
    a.click();
    window.URL.revokeObjectURL(url);
}

startStream();
recordButton.addEventListener('click', startRecording);
stopRecordButton.addEventListener('click', stopRecording);


let recording = false;
recordButton.addEventListener('click', () => {
    if (!recording) {
        // Bắt đầu ghi video
        recording = true;

        // Thay đổi màu nền của nút "Record"
        recordButton.style.backgroundColor = '#7fdbff';
        stopRecordButton.style.backgroundColor = '#ff4136';

        // Đổi nội dung của nút "Record"
        recordButton.textContent = 'Recording...';

        // Thực hiện các hành động bắt đầu ghi video
    } else {
        // Dừng ghi video
        recording = false;

        // Trả lại màu nền ban đầu của nút "Record"
        recordButton.style.backgroundColor = '#0074d9';
        stopRecordButton.style.backgroundColor = '#ff4136';

        // Trả lại nội dung ban đầu của nút "Record"
        recordButton.textContent = 'Start Record';

        // Thực hiện các hành động dừng ghi video
    }
});

stopRecordButton.addEventListener('click', () => {
    // Thay đổi màu nền của nút "Stop Record"
    stopRecordButton.style.backgroundColor = '#ff4136';

    // Thực hiện các hành động khi nhấp vào nút "Stop Record"
    // (ví dụ: dừng ghi video)
});


function uploadFile() {
    var formData = new FormData();
    var fileInput = document.getElementById('file');
    formData.append('file', fileInput.files[0]);

    var xhr = new XMLHttpRequest();

    xhr.open('POST', '/upload', true);

    xhr.onload = function() {
        if (xhr.status === 200) {
            
            document.getElementById('result').value = xhr.responseText;
        } else {
            document.getElementById('result').innerHTML = "Có lỗi xảy ra khi tải lên tệp.";
        }
    };

    xhr.send(formData);
}