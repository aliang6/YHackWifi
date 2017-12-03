$(function(){
	var recommend = "We recommend the ";
	(% plan %)
		recommend += plan + " plan.";
	(% endif %)

	(% userPrices %)
		var prices = [userPrices["BRONZE"], userPrices["SILVER"], userPrices["GOLD"], userPrices["PLATINUM"]];
		d3.csv("https://raw.githubusercontent.com/syntagmatic/parallel-coordinates/master/examples/data/nutrients.csv", function(data) {
            var colorgen = d3.scale.ordinal()
            .range(["#a6cee3","#1f78b4","#b2df8a","#33a02c",
                    "#fb9a99","#e31a1c","#fdbf6f","#ff7f00",
                    "#cab2d6","#6a3d9a","#ffff99","#b15928"]);

            var color = function(d) { return colors(d.group); };

            var parcoords = d3.parcoords()("#example-progressive")
            .data(data)
            .hideAxis(["name"])
            .color(color)
            .alpha(0.25)
            .composite("darken")
            .margin({ top: 24, left: 150, bottom: 12, right: 0 })
            .mode("queue")
            .render()
            .brushMode("1D-axes");
            parcoords.highlight(prices);
            parcoords.svg.selectAll("text")
            .style("font", "10px sans-serif");
        });
	(% endif %)
	$("#fade").css("display", "none");
	$("#fade").fadeIn(1000);
	$("#submit").click(function(){
		cust_name = $("#name_box").val();
		$("#fade:visible").fadeOut(1000, function(){
			window.location = "index.html";
		});
	});
	$("#recommend").css({margin-top: 2%;, font-size: 10px;}
});