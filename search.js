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

function store() {
	cust_name = getElementsByName("cust_name");
	sex = getElementsByName("sex");
	DOB = getElementsByName("DOB");
	state = getElementsByName("state");
	city = getElementsByName("city");
	var json = search();
	document.getElementsByName("json").innerHTML = json;
}

getElementsByName("Submit").onclick = 
	function search() {
		//Search Participants and retrieve the Participant name, date of birth and gender
		//SELECT pi.name, pi.DOB, pi.sex FROM v_participant pi WHERE pi.state = 'Alaska'
		url = "https://v3v10.vitechinc.com/solr/v_participant/select?indent=on&wt=json" + "&q=state:" + state + "+sex:" + sex "&rows=100" + "&fl=name,DOB,sex,state";
		fetch(url)
			.then(res => res.json())
			.then(body => alert(body.response.numFound))
			.catch(alert)
	}