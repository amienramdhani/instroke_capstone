const toogle = document.querySelector('.menu');
const navList = document.querySelector('#unordered-list');

toogle.addEventListener('click', () => {
  navList.classList.toggle('active');
});

document.querySelectorAll('a[href^="#"]').forEach((anchor) => {
  anchor.addEventListener('click', function (e) {
    e.preventDefault();
    document.querySelector(this.getAttribute('href')).scrollIntoView({
      behavior: 'smooth',
    });
  });
});
