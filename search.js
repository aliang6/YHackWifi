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

function fadeOut(elements) {
    var alpha = 1;  // initial opacity
    var timer = setInterval(function () {
        if (elements.style.opacity <= 0.1){
            clearInterval(timer);
            elements.style.display = 'none';
        }
        elements.style.opacity = alpha;
        alpha -= alpha * 0.1;
    }, 50);
}

function getName() {
	cust_name = document.getElementById("cust_name").value;
	fadeOut(document.getElementById("fade"));
}

document.getElementById("submit").addEventListener("click", getName());

function store() {
	sex = document.getElementById("sex").value;
	DOB = document.getElementById("DOB").value;
	state = document.getElementById("state").value;
	city = document.getElementById("city").value;
	var json = JSON.parse(search());
	document.innerHTML = "Your mom"
	document.innerHTML = json;
	//document.getElementsById("json").innerHTML = json;
}

//document.getElementById("submit").addEventListener("click", function(){ alert("Your mom"); });


function search() {
	//Search Participants and retrieve the Participant name, date of birth and gender
	//SELECT pi.name, pi.DOB, pi.sex FROM v_participant pi WHERE pi.state = 'Alaska'
	url = "https://v3v10.vitechinc.com/solr/v_participant/select?indent=on&wt=json" + "&q=state:" + state + "+sex:" + sex + "&rows=100" + "&fl=name,DOB,sex,state";
	fetch(url, {mode: 'cors'})
		.then(res => res.json())
		.then(body => alert(body.response.numFound))
		.catch(alert)
}