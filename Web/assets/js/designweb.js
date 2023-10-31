//transitions
document.body.classList.add('fade-in');

//darkmode
const themeToggle = document.getElementById('theme');
const body = document.body;

themeToggle.onclick = () => {
  body.classList.toggle('dark-mode');
  themeToggle.classList.toggle('fa-moon');
  themeToggle.classList.toggle('fa-sun');
}




//height of box
window.onload = function() {
    var video = document.getElementById('video');
    var resultFrame = document.getElementById('result-frame');

    video.onloadedmetadata = function() {
        resultFrame.style.height = video.offsetHeight + 'px';
    };
};

//popup
const guideButton = document.querySelector('.home-button:nth-child(2)');
const popup = document.querySelector('.popup');
const popupExitButton = document.querySelector('.popup-exit');

guideButton.addEventListener('click', () => {
    popup.style.display = 'block';
});

popupExitButton.addEventListener('click', () => {
    popup.style.display = 'none';
});
  
