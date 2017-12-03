$(function(){
	$("#fade").css("display", "none");
	$("#fade").fadeIn(1000);
	$("#submit").click(function(){
		cust_name = $("#name_box").val();
		$("#fade:visible").fadeOut(1000, function(){
			window.location = "index.html";
		});
	});
	var probability = Math.random() * 100;
	if(probability >= 0 && probability <= 20){
		$("#recommend").html("We recommend the bronze plan.")
	}
	else if(probability > 20 && probability <= 60){
		$("#recommend").html("We recommend the silver plan.")
	}
	else if(probability > 60 && probability <= 90){
		$("#recommend").html("We recommend the gold plan.")
	}
	else{
		$("#recommend").html("We recommend the platinum plan.")
	}
	$("#recommend").css({margin-top: 2%;, font-size: 10px;})
});