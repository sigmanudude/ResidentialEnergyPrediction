// Main javascript file that will invoke all the function to populate the dashboard

// ########### Declare global variables
var predictPriceURL = "/predict/"

var submitBtn = d3.select("#filter-btn"); 
var clearBtn = d3.select("#clear-filter-btn"); 
// Grab a reference to the dropdown select element
var regSel = d3.select("#region");
// var distSel = d3.select("#district");
var sqftSel = d3.select("#sqft");

// grab instance of div to display data
var tblDiv = d3.select("#dataTbl");
var dataBtn = d3.select("#data-btn");
var pElement = d3.select("#pageDetails");
var prev = d3.select("#data-btn-prev");
var next = d3.select("#data-btn-next");
var last = d3.select("#data-btn-last");
var first = d3.select("#data-btn-first");
var distSumElement = d3.select("#distSum");

var startpg = 1, totalpg = 36, currpg = 1;

var spinner_HTM = `<div class="d-flex justify-content-center"><div class="spinner-border" style="width: 3rem; height: 3rem;" role="status"><span class="sr-only">Predicting...</span></div></div>`


// function that initiliazes the page
function init(){
    // populate the dropdown filters
    // populateFilters();
    
    //attach event to submit buttons
    // associate event to the buttons
    submitBtn.on("click", function(){filterData();});
    clearBtn.on("click", function(){resetFilters();});
    // dataBtn.on("click", function(){displayData(_yr,_cat,_dist,tblDiv, currpg);});
    // prev.on("click", function(){displayData(_yr,_cat,_dist,tblDiv,currpg-1);});
    // next.on("click", function(){displayData(_yr,_cat,_dist,tblDiv,currpg+1);});
    // first.on("click", function(){displayData(_yr,_cat,_dist,tblDiv,startpg);});
    // last.on("click", function(){displayData(_yr,_cat,_dist,tblDiv,totalpg);});

    
}

init();

// Helper functions
function onlyUnique(value, index, self) { 
    return self.indexOf(value) === index;
}

function boxPlot_byYr(){
    d3.json(`/boxPlot`).then(function(pltdata){
        // console.log(pltdata)
        var yr = pltdata.map(r => +r.Year)
        var unqYr = yr.filter( onlyUnique )
        // console.log(unqYr);
        // declare data array for box plots
        data = [];
        unqYr.map(yr => {
            data.push({"y" : pltdata.filter(function(r){
                            return +r.Year === yr;
                        }).map(c => +c.Cnt), 
                        "type": "box",
                        "name" : yr.toString()
                });
        });
        console.log(data);

        var layout = {
            // title: 'Variance of Mean of Violation over years',
            autosize: true,
            height:200,
            margin: {
                l: 5,
                r: 5,
                b: 20,
                t: 10,
                pad: 4
            },
            font:{size:10},
            
            showlegend:false
          };
        Plotly.newPlot("boxWhisker", data, layout,{displayModeBar: false, responsive: true});
    });
};

function populateFilters() {    
    // Use the list of sample names to populate the select options
    d3.json("/filterData").then((filtData) => {
        console.log(filtData['Sample']);
        filtData['Sample'].forEach((item) => {
        sampSel
          .append("option")
          .text(`${item}`)
          .property("value", item.split(" ")[1]);
        console.log(item);
      });      
    });
  };

  

function filterData(){
    d3.event.preventDefault();
    
    var regionVal = regSel.property("value");
    var sqftVal = sqftSel.property("value");
    
    d3.select("#resPredict").html(spinner_HTM);
    console.log(`${regionVal}, ${sqftVal}`);

     d3.json(`${predictPriceURL}${regionVal}/${sqftVal}`).then(function(data){
    //     // remove already displayed
    console.log(data['PredictResults'])
        
    //    print the table sent by Flask
         d3.select("#resPredict").html(data['PredictResults']); // Print the prediction selected
         d3.select("#dataTbl").html(data['metadata']); // print the sample data
    //     Object.entries(data).forEach(([key, value]) => {
    //       if(key !== "sample" && key !== "WFREQ")
    //         metadata_panel.append("span")
    //                     .attr("style","font-size: 11px;")
    //                     .text(`${key} : ${value} | `);
    //         });
    });    
};

function resetFilters(){
    d3.event.preventDefault();
    sampSel.selectAll("option").property("selected",function(d){ return d === 0; })
    modelSel.selectAll("option").property("selected",function(d){ return d === "all"; })
    // distSel.selectAll("option").property("selected",function(d){ return d === 0; })
    
    // // redraw map features layer All data
    // addEdit_MapLayers(_yr,_cat,_dist, "update");

    // // reset all barplots
    // dynBarPlots(_yr,_cat,_dist, 'violationByDist', "distSpread", "Districts");
    // dynBarPlots(_yr,_cat,_dist,'violationByCat', "violationCat","Categories");
    // // dynBarPlots(_yr,_cat,_dist,'violationByType', "violationType");
};







function displayData(y,c,d,divElement,pg){
    d3.event.preventDefault();
    pg <= 0? pg = 1: pg > 36 ? pg = 36 : pg = pg;

    d3.json(`/dashboardData/${y}/${c}/${d}/${pg}`).then(function(data){
        // console.log(data.html);
        pElement.node().innerHTML = `Displaying page ${pg} of ${totalpg}`
        divElement.node().innerHTML = "";
        divElement.node().innerHTML = data.html;

    }); // end of JSON
};

