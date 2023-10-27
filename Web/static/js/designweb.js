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




//height of textarea
window.addEventListener('load', function() {
    var img = document.getElementById('detect-img');
    var frame = document.getElementById('result-frame');
    frame.style.height = img.clientHeight + 'px';
});

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
  
