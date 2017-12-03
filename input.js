// Participants
var cust_name, DOB, state, city, sex;
// Participant Details
var EMPLOYMENT_STATUS, ANNUAL_INCOME, MARITAL_STATUS, HEIGHT, WEIGHT, TOBACCO, PEOPLE_COVERED, OPTIONAL_INSURED;
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
		$("#fade:visible").fadeOut(1000, function(){
			/*DOB = $("#dob").val();
			state = $("#state").val();
			city = $("#city").val();
			EMPLOYMENT_STATUS = $("#emp_status").val();
			ANNUAL_INCOME = $("#ann_inc").val();
			MARITAL_STATUS = $("#mar_stat").val();
			HEIGHT = $("#height").val();
			WEIGHT = $("#weight").val();
			TOBACCO = $("#tobacco").val();
			if($('#hepBy').prop('checked')){
				hepatitis_B = 1;
			}
			else{
				hepatitis_B = 0;
			}

			if($('#acy').prop('checked')){
				ataxic_cerebral = 1;
			}
			else{
				ataxic_cerebral = 0;
			}

			if($('#diary').prop('checked')){
				diarrhea = 1;
			}
			else{
				diarrhea = 0;
			}

			if($("#tachy").prop('checked')){
				tachcardia = 1;
			}
			else{
				tachcardia = 0;
			}

			if($("#osay").prop('checked')){
				sleep_apnea = 1;
			}
			else{
				sleep_apnea = 0;
			}

			if($("#fmbly").prop('checked')){
				metacarpal_fracture = 1;
			}
			else{
				metacarpal_fracture = 0;
			}


			if($("#ahby").prop('checked')){
				heart_abnormalities = 1;
			}
			else{
				heart_abnormalities = 0;
			}

			if($("#t2dy").prop('checked')){
				diabetes = 1;
			}
			else{
				diabetes = 0;
			}

			if($("#hivy").prop('checked')){
				HIV = 1;
			}
			else{
				HIV = 0;
			}

			if($("#cwhy").prop('checked')){
				hemorrhage_cough = 1;
			}
			else{
				hemorrhage_cough = 0;
			}*/
			window.location = "./par_coor";
		});
	});
});