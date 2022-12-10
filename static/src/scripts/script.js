// event pada saat link diklik
$('.page-scroll').on('click', function(e) {

	// ambil isi href 
	var tujuan = $(this).attr('href');
	// tangkap elemen ybs
	var elemenTujuan = $(tujuan);

		// pindahkan scroll
		$('html,body').animate({
			scrollTop:elemenTujuan.offset().top - 70
		}, 1250, 'swing');

		e.preventDefault();

});

$('.custom-file-input').on('change', function() { 
	let fileName = $(this).val().split('\\').pop(); 
	$(this).next('.custom-file-label').addClass("selected").html(fileName); 
});



// // parallax


// $(window).scroll(function(){

// 	var wScroll = $(this).scrollTop();

// 	// jumbotron

// 	$('.jumbotron img').css({
// 		'transform' : 'translate(0px, '+ wScroll/6 +'%)'
// 	});	

// 	$('.jumbotron h1').css({
// 		'transform' : 'translate(0px, '+ wScroll/1.6 +'%)'
// 	});	

// 	$('.jumbotron p').css({
// 		'transform' : 'translate(0px, '+ wScroll/1 +'%)'
// 	});	

// 	// about
// 	if( wScroll > $('.about').offset().top - 150){	
// 		$('.about .pKiri').addClass('pMuncul');
// 		$('.about .pKanan').addClass('pMuncul');
// 	}
// 	// portfolio

// 	if( wScroll > $('.portfolio').offset().top - 150){
// 		$('.portfolio .thumbnail').each(function(i) {
// 			setTimeout(function() {
// 				$('.portfolio .thumbnail').eq(i).addClass('muncul');		
// 			} ,200 * (i+1));
// 		});


					
// 	}

// });

