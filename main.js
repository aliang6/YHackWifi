// Participants
var cust_name, DOB, state, city, sex;
// Participant Details
var EMPLOYMENT_STATUS, ANNUAL_INCOME, MARITAL_STATUS, HEIGHT, WEIGHT, TOBACCO;
// Participant Details Preconditions
var hepatitis_B, ataxic_cerebral, diarrhea, tachycardia, sleep_apnea, metacarpal_fracture, heart_abnormalities, diabetes, HIV, hemorrhage_cough;
// Quotes
var PLATINUM, GOLD, SILVER, BRONZE, PURCHASED;
// Plan Details
var BASE_PRICE, AD_D, DEDUCTIBLE, PLAN_NAME, COVERAGE_AMOUNT;

//JSON
var json;

$(function(){
	$("#fade").css("display", "none");
	$("#fade").fadeIn(1000);
	$("#submit").click(function(){
		cust_name = $("#name_box").val();
		$("#fade:visible").fadeOut(1000, function(){
			window.location = "input.html";
		});
	});
});