var host = "lahorekid.github.io";
if ((host == window.location.host) && (window.location.protocol != "https:"))
   window.location.protocol = "https";

if (getCookie("theme") == "dark"){
	$('html').addClass("dark");
}

var nextFeedPost = 20;
var postList;
var triggered = false;
var jsonRetrieved = false;

$(document).ready(function(){
	var topOfContent = $(".page-content").offset().top;
	var windowHeight = $(window).height();
	displayImages(windowHeight);
	$('.menu-icon').click(function(event){
		event.preventDefault();
		$('.trigger').toggleClass('active');
	});
	$('.page-content').click(function(event){
		$('.trigger').removeClass('active');
		$(".right-nav .search-form").slideUp(300, function(){$("#search-icon").fadeIn(100)});
	})
	$('.post img').not('.featured-image').click(function(event){
		window.location.href = $(this).attr("src");
	})
	
	$(window).scroll(function() {
        var imageTrigger = windowHeight + $(window).scrollTop();

        var postListTrigger = $(window).scrollTop() + windowHeight;
        var postListTrigger = $(document).height() - 1000;
        if (!triggered && (imageTrigger > postListTrigger) && $("#home").length){
	  		triggered = true;
        	if(!jsonRetrieved){
				jsonRetrieved = true;
	        	var request = $.ajax({
	        		method: "GET",
	        		url: '/assets/js/all-posts.json'
	        	});
	        	request.success(function (data) {
					postList = $('.feed-item', '<div>' + data + '</div>');
					postList = data.all_posts;
					appendFeedContent();
				});
				request.fail(function(data){
					console.log("Failed to retrieve more posts.");
				})
			}
			else{
				appendFeedContent();
			}
        }
    	displayImages(imageTrigger);
	});
	$(".menu-icon").click(function(e){
		e.preventDefault();
		$(".site-nav").toggleClass("menu-toggled");
	});
	$("#search-icon").click(function(e){
		e.preventDefault();
		$(this).fadeOut(100);
		$(".right-nav .search-form").slideDown(300);
		$(".right-nav .search-form .search-box").focus();
	});
	$(".theme-toggle").click(function(e){
		e.preventDefault();
		$('html').toggleClass("dark");
		if ($('html').hasClass("dark")){
			document.cookie = "theme=dark;path=/";
		}
		else {
			document.cookie = "theme=light;path=/";
		}
	})
});

function appendFeedContent(){
	if (postList[nextFeedPost]){
		$('.posts').append($('<div>' + postList[nextFeedPost] + '</div>').text());
		nextFeedPost++;
	}
	triggered = false;
}
function displayImages(loc){
	var imgs = $(".feed-pic .hidden");
	for (var i = 0; i<imgs.length; i++){
		if (imgs[i].offsetTop < (loc - 50)){
			$(imgs[i]).fadeOut(0, function(){
				$(this).attr("src", $(imgs[i]).attr("data-src")).fadeIn(600);
			    $(imgs[i]).attr("data-src", "");
			    $(imgs[i]).removeClass("hidden");
			    $(imgs[i]).addClass("visible");
		    });
		}
		else{
			return;
		}
	}
}

function getCookie(cname) {
    var name = cname + "=";
    var ca = document.cookie.split(';');
    for(var i = 0; i <ca.length; i++) {
        var c = ca[i];
        while (c.charAt(0)==' ') {
            c = c.substring(1);
        }
        if (c.indexOf(name) == 0) {
            return c.substring(name.length,c.length);
        }
    }
    return "";
}
