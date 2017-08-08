//  Performance Chart

zingchart.MODULESDIR = "https://cdn.zingchart.com/modules/";
ZC.LICENSE = ["569d52cefae586f634c54f86dc99e6a9", "ee6b7db5b51705a13dc2339db3edaf6d"];

var tab_value_index = []

$(document).on('click', '.chart-button', function(event) {

    var get_all_tab='';
    var ij = 0;

    $( ".chart-pane-tab" ).each(function() {
        get_all_tab= $(".chart-tab-icon-text" ).text();

    });

    arr = get_all_tab.split('K'); 

    for(i=0; i < arr.length; i++)
        tab_value_index[i] = arr[i];

});
function indexOf(arr) {
	var len = arr.length;
	var indices = new Array(len);
	for (var i = 0; i < len; ++i) indices[i] = i;
	indices.sort(function (a, b) { return arr[a] < arr[b] ? -1 : arr[a] > arr[b] ? 1 : 0; });

    return indices.reverse();
}

function sortObject(obj) {
    var arr = [];
    var prop;
    for (prop in obj) {
        if (obj.hasOwnProperty(prop)) {
            arr.push({
                'key': prop,
                'value': obj[prop]
            });
        }
    }
    arr.sort(function(a, b) {
        return b.value - a.value;
    });
    return arr; // returns array
}

function rank_index(str)
{
	var search1 = str.search("k");
	var res1 = str.substr(search1+2);

	return res1;
}

function day1(test)
{
	$(".chart_loader").show();

    var board_value = $('.chart-title-text').html();
    board_value=board_value.split(":");
    board_value=board_value[1].replace(/\s+/g, '');

    var tab_value=$('.chart-pane-tab-on').text();
    tab_value = tab_value.replace(/\s+/g, '');

    var anti_val = '';

    $.ajax
    ({
        url: '/getchartdata',
        type: 'get',
        dataType: 'json', // added data type
        success: function(result)
        {

          /* ------------------
          ---------------------
          Ranking_chart
          ---------------------
          --------------------- */

          $(".chart_loader").hide();

          var date = [];
          var system = [];
          var anti_system = [];
          var benchmark = [];
          var cumper = [];

          var anti_system_cum = [];
          var benchmark_cum = [];

          var chart_title = '';
          
  			/* ------------------
            ---------------------
            Ranking Chart
            ---------------------
            --------------------- */
      
            var rank = [];
            var value_1 = [];
            var value_2 = [];
            var value_3 = [];
            btn_click = 0;
            // var date_len = date.length-1;

            if(tab_value==tab_value_index[2] + "K")
            {
                $.each(result.v4micro_performance, function(l,m) {
                  if(l==board_value)
                  {
                      $.each(this, function(k, v) {

                          date.push(k);
                          system.push(v);

                      });
                  }
                });
                var rank_title = "v4micro " + board_value + " ‘Ranking from’ " + date[0] + " to " + date[date.length-1]; 

                if(test==1)
                {
                	btn_click = 1;
                    var day1 = sortObject(result.v4micro_ranking['1Day Lookback']);

                    var day5 = result.v4micro_ranking['3Day Lookback'];

                    var day20 = result.v4micro_ranking['20Day Lookback'];

                    $.each(day1, function(l,m) {

                        var vv=0;
                        $.each(this, function(k, v) {

                            if(k!='-8 Rank RiskOff' && k!='9 Rank RiskOn')
                            {


                                if(vv==0)
                                {
                                    rank.push(v);
                                    vv=vv+1;

                                }
                                else
                                {

                                    value_3.push(v);
                                    vv=0;                                    

                                }   
                            }
                        });
                    });

                 
                    $.each(rank, function(k, v) {
                           
                        $.each(day5, function(o, p) {
                        	if(k!='-8 Rank RiskOff' && k!='9 Rank RiskOn')
                            {

	                            if(o==v)
	                            {
                                  	value_2.push(p);
                                }

                            }
                        });

                        $.each(day20, function(o, p) {
                        	if(k!='-8 Rank RiskOff' && k!='9 Rank RiskOn')
                            {

	                            if(o==v)
	                            {
                                	value_1.push(p);
								}
                            }

                        });
                    });
                }
                 else if(test==5)
                {
                	btn_click = 2;
                    var day5 = sortObject(result.v4micro_ranking['3Day Lookback']);

                    var day20 = result.v4micro_ranking['20Day Lookback'];

                    var day1 = result.v4micro_ranking['1Day Lookback'];

                    $.each(day5, function(l,m) {

                        var vv=0;
                        $.each(this, function(k, v) {

                            if(k!='-8 Rank RiskOff' && k!='9 Rank RiskOn')
                            {
                                if(vv==0)
                                {
                                    
                                    rank.push(v);
                                    vv=vv+1;

                                }
                                else
                                {
                                    
                                    value_2.push(v);
                                    vv=0;

                                }   
                            }
                        });
                    });

                 
                    $.each(rank, function(k, v) {
                           
                        $.each(day20, function(o, p) {

                        	if(k!='-8 Rank RiskOff' && k!='9 Rank RiskOn')
                            {

	                            if(o==v)
	                            {
	                                
	                                value_1.push(p);

	                            }
	                        }
                        });

                        $.each(day1, function(o, p) {
                        	if(k!='-8 Rank RiskOff' && k!='9 Rank RiskOn')
                            {

	                            if(o==v)
	                            {
	                                
	                                value_3.push(p);

	                            }
	                        }

                        });
                    });
                }

                else if(test==20)
                {
                	btn_click = 3;
                    var day20 = sortObject(result.v4micro_ranking['20Day Lookback']);

                    var day5 = result.v4micro_ranking['3Day Lookback'];

                    var day1 = result.v4micro_ranking['1Day Lookback'];

                    $.each(day20, function(l,m) {

                        var vv=0;
                        $.each(this, function(k, v) {

                            if(k!='-8 Rank RiskOff' && k!='9 Rank RiskOn')
                            {
                                if(vv==0)
                                {
                                    
                                    rank.push(v);
                                    vv=vv+1;

                                }
                                else
                                {
                                    
                                    value_1.push(v);
                                    vv=0;

                                }   
                            }
                        });
                    });

                 
                    $.each(rank, function(k, v) {
                           
                        $.each(day5, function(o, p) {
                        	if(k!='-8 Rank RiskOff' && k!='9 Rank RiskOn')
                            {

	                            if(o==v)
	                            {

	                                value_2.push(p);

	                            }
	                        }
                        });

                        $.each(day1, function(o, p) {
                        	if(k!='-20 Rank RiskOff' && k!='20 Rank RiskOn')
                            {

	                            if(o==v)
	                            {

	                                
	                                  value_3.push(p);

	                            }
	                        }

                        });
                    });
                }
            }
            else if(tab_value==tab_value_index[1] + "K")
            {
                $.each(result.v4mini_performance, function(l,m) {
                  if(l==board_value)
                  {
                      $.each(this, function(k, v) {

                          date.push(k);
                          system.push(v);

                      });
                  }
                });
                var rank_title = "v4mini " + board_value + " ‘Ranking from’ " + date[0] + " to " + date[date.length-1]; 

                if(test==1)
                {
                	btn_click = 1;
                    var day1 = sortObject(result.v4mini_ranking['1Day Lookback']);

                    var day5 = result.v4mini_ranking['3Day Lookback'];

                    var day20 = result.v4mini_ranking['20Day Lookback'];

                    $.each(day1, function(l,m) {

                        var vv=0;
                        $.each(this, function(k, v) {

                            if(k!='-4 Rank RiskOff' && k!='4 Rank RiskOn')
                            {
                                if(vv==0)
                                {
                                    
                                    rank.push(v);
                                    vv=vv+1;

                                }
                                else
                                {
                                    
                                    value_3.push(v);
                                    vv=0;

                                }   
                            }
                        });
                    });

                 
                    $.each(rank, function(k, v) {
                           
                        $.each(day5, function(o, p) {
                        	if(k!='-4 Rank RiskOff' && k!='4 Rank RiskOn')
                            {

	                            if(o==v)
	                            {

	                                
	                                  value_2.push(p);

	                            }
	                        }
                        });

                        $.each(day20, function(o, p) {
                        	if(k!='-4 Rank RiskOff' && k!='4 Rank RiskOn')
                            {

	                            if(o==v)
	                            {

	                                
	                                  value_1.push(p);

	                            }
	                        }

                        });
                    });
                }
                 else if(test==5)
                {
                	btn_click = 2;
                    var day5 = sortObject(result.v4mini_ranking['3Day Lookback']);

                    var day20 = result.v4mini_ranking['20Day Lookback'];

                    var day1 = result.v4mini_ranking['1Day Lookback'];

                    $.each(day5, function(l,m) {

                        var vv=0;
                        $.each(this, function(k, v) {

                            if(k!='-4 Rank RiskOff' && k!='4 Rank RiskOn')
                            {
                                if(vv==0)
                                {
                                    
                                    rank.push(v);
                                    vv=vv+1;

                                }
                                else
                                {
                                    
                                    value_2.push(v);
                                    vv=0;

                                }   
                            }
                        });
                    });

                 
                    $.each(rank, function(k, v) {
                           
                        $.each(day20, function(o, p) {
                        	if(k!='-4 Rank RiskOff' && k!='4 Rank RiskOn')
                            {

	                            if(o==v)
	                            {

	                                
	                                  value_1.push(p);

	                            }
	                        }
                        });

                        $.each(day1, function(o, p) {
                        	if(k!='-4 Rank RiskOff' && k!='4 Rank RiskOn')
                            {

	                            if(o==v)
	                            {

	                                
	                                  value_3.push(p);

	                            }
	                        }

                        });
                    });
                }

                else if(test==20)
                {
                	btn_click = 3;
                    var day20 = sortObject(result.v4mini_ranking['20Day Lookback']);

                    var day5 = result.v4mini_ranking['3Day Lookback'];

                    var day1 = result.v4mini_ranking['1Day Lookback'];

                    $.each(day20, function(l,m) {

                        var vv=0;
                        $.each(this, function(k, v) {

                            if(k!='-4 Rank RiskOff' && k!='4 Rank RiskOn')
                            {
                                if(vv==0)
                                {
                                    
                                    rank.push(v);
                                    vv=vv+1;

                                }
                                else
                                {
                                    
                                    value_1.push(v);
                                    vv=0;

                                }   
                            }
                        });
                    });

                 
                    $.each(rank, function(k, v) {
                           
                        $.each(day5, function(o, p) {
                        	if(k!='-4 Rank RiskOff' && k!='4 Rank RiskOn')
                            {

	                            if(o==v)
	                            {

	                                
	                                  value_2.push(p);

	                            }
	                        }
                        });

                        $.each(day1, function(o, p) {
                        	if(k!='-4 Rank RiskOff' && k!='4 Rank RiskOn')
                            {

	                            if(o==v)
	                            {

	                                
	                                  value_3.push(p);

	                            }
	                        }

                        });
                    });
                }
            }
            
            else if(tab_value==tab_value_index[0] + "K")
            {
                $.each(result.v4mini_performance, function(l,m) {
                  if(l==board_value)
                  {
                      $.each(this, function(k, v) {

                          date.push(k);
                          system.push(v);

                      });
                  }
                });   
                var rank_title = "v4futures " + board_value + " ‘Ranking from’ " + date[0] + " to " + date[date.length-1]; 

                if(test==1)
                {
                	btn_click = 1;
                    var day1 = sortObject(result.v4futures_ranking['1Day Lookback']);

                    var day5 = result.v4futures_ranking['3Day Lookback'];

                    var day20 = result.v4futures_ranking['20Day Lookback'];

                    $.each(day1, function(l,m) {

                        var vv=0;
                        $.each(this, function(k, v) {

                            if(k!='-20 Rank RiskOff' && k!='20 Rank RiskOn')
                            {
                                if(vv==0)
                                {
                                    
                                    rank.push(v);
                                    vv=vv+1;

                                }
                                else
                                {
                                    
                                    value_3.push(v);
                                    vv=0;

                                }   
                            }
                        });
                    });

                 
                    $.each(rank, function(k, v) {
                           
                        $.each(day5, function(o, p) {
                        	if(k!='-20 Rank RiskOff' && k!='20 Rank RiskOn')
                            {

	                            if(o==v)
	                            {

	                                
	                                  value_2.push(p);

	                            }
	                        }
                        });

                        $.each(day20, function(o, p) {
                        	if(k!='-20 Rank RiskOff' && k!='20 Rank RiskOn')
                            {

	                            if(o==v)
	                            {

	                                
	                                  value_1.push(p);

	                            }
	                        }

                        });
                    });
                }
                 else if(test==5)
                {
                	btn_click = 2;
                    var day5 = sortObject(result.v4futures_ranking['3Day Lookback']);

                    var day20 = result.v4futures_ranking['20Day Lookback'];

                    var day1 = result.v4futures_ranking['1Day Lookback'];

                    $.each(day5, function(l,m) {

                        var vv=0;
                        $.each(this, function(k, v) {

                            if(k!='-20 Rank RiskOff' && k!='20 Rank RiskOn')
                            {
                                if(vv==0)
                                {
                                    
                                    rank.push(v);
                                    vv=vv+1;

                                }
                                else
                                {
                                    
                                    value_2.push(v);
                                    vv=0;

                                }   
                            }
                        });
                    });

                 
                    $.each(rank, function(k, v) {
                           
                        $.each(day20, function(o, p) {
                        	if(k!='-20 Rank RiskOff' && k!='20 Rank RiskOn')
                            {

	                            if(o==v)
	                            {

	                                
	                                  value_1.push(p);

	                            }
	                        }
                        });

                        $.each(day1, function(o, p) {
                        	if(k!='-20 Rank RiskOff' && k!='20 Rank RiskOn')
                            {

	                            if(o==v)
	                            {

	                                
	                                  value_3.push(p);

	                            }
	                        }

                        });
                    });
                }

                else if(test==20)
                {
                	btn_click = 3;
                    var day20 = sortObject(result.v4futures_ranking['20Day Lookback']);

                    var day5 = result.v4futures_ranking['3Day Lookback'];

                    var day1 = result.v4futures_ranking['1Day Lookback'];

                    $.each(day20, function(l,m) {

                        var vv=0;
                        $.each(this, function(k, v) {

                            if(k!='-20 Rank RiskOff' && k!='20 Rank RiskOn')
                            {
                                if(vv==0)
                                {
                                    
                                    rank.push(v);
                                    vv=vv+1;

                                }
                                else
                                {
                                    
                                    value_1.push(v);
                                    vv=0;

                                }   
                            }
                        });
                    });

                 
                    $.each(rank, function(k, v) {
                           
                        $.each(day5, function(o, p) {
                        	if(k!='-20 Rank RiskOff' && k!='20 Rank RiskOn')
                            {

	                            if(o==v)
	                            {

	                                
	                                  value_2.push(p);

	                            }
	                        }
                        });

                        $.each(day1, function(o, p) {
                        	if(k!='-20 Rank RiskOff' && k!='20 Rank RiskOn')
                            {

	                            if(o==v)
	                            {

	                                
	                                  value_3.push(p);

	                            }
	                        }

                        });
                    });
                }
            }

            // console.log("test");
            var data  = [];
            var i = 0;
            count = 0;
			identify_color = "#000000";
            value_1_color = "#0000FF";
            value_2_color = "#F7C143";
            value_3_color = "#0B850C";

            // console.log(board_value);

            $.each(rank, function(k,v) {

            	i_color = "";
            	identify_color = "#000000";
            	value_1_color = "#0000FF";
	            value_2_color = "#F7C143";
	            value_3_color = "#0B850C";

            	v = rank_index(v);
            	// console.log(v);

            	if(board_value == v   ||   "Anti-" + board_value == v)
            	{
            		i_color = "#FF0000";
            		if(btn_click == 1)
            		{
            			value_3_color = "#FF0000";
            		}
            		else if(btn_click == 2)
            		{
            			value_2_color = "#FF0000";
            		}
            		else if(btn_click == 3)
            		{
            			value_1_color = "#FF0000";
            		}
            	}

            	if(v ==  "Previous"  	||  v ==  "Anti-Previous"  	|| v ==  "LowestEquity"	||  v ==  "AntiLowestEquity"||
            	   v ==  "RiskOff"		||  v ==  "RiskOn"  		|| v ==  "Custom"		||  v ==  "Anti-Custom"  	||
            	   v ==  "Seasonality"	||  v ==  "Anti-Seasonality"|| v ==  "HighestEquity"||  v ==  "AntiHighestEquity"||
            	   v ==  "50/50"		||  v ==  "Anti50/50"  		|| v ==  "Off"			||  v ==  "on"  			||
            	   v ==  "benchmark"  )
            	{
            		i_color = "#0000FF";
            	}

            	if(btn_click == 1)
            	{
            		if(value_3[i] == value_3[i-1])
            		{
            			v = v + " (" + count + ")";
            		}
            		else
            		{
            			count = count + 1;
            			v = v + " (" + count + ")";            			
            		}
            	}
            	else if(btn_click == 2)
            	{
            		if(value_2[i] == value_2[i-1])
            		{
            			v = v + " (" + count + ")";
            		}
            		else
            		{
            			count = count + 1;
            			v = v + " (" + count + ")";            			
            		}
            	}
            	else if(btn_click == 3)
            	{
            		if(value_1[i] == value_1[i-1])
            		{
            			v = v + " (" + count + ")";
            		}
            		else
            		{
            			count = count + 1;
            			v = v + " (" + count + ")";            			
            		}
            	}

        		v1 = Math.abs(value_1[i])
        		v2 = Math.abs(value_2[i])
        		v3 = Math.abs(value_3[i])

            	var arr = [v1,v2,v3];

				var check = indexOf(arr);

				if(check[0] == 0 && check[1] == 1 && check[2] == 2 )
				{
					if(i_color !== '')
					{
						identify_color = i_color;
					}

					data.push({
						"category" : v,
						"color" : identify_color,
						"segments" : [{
							"start" : 0,
							"end" : value_1[i],
							"colors" : value_1_color,
							"task": "20Day"
						},{
							"start" : 0,
							"end" : value_2[i],
							"colors" : value_2_color,
							"task": "3Day"
						},{
							"start" : 0,	
							"end" : value_3[i],
							"colors" : value_3_color,
							"task": "1Day"
						}]
					});
					i_color = '';
				}

				else if(check[0] == 0 && check[1] == 2 && check[2] == 1 )
				{
					if(i_color !== '')
					{
						identify_color = i_color;
					}

					data.push({
						"category" : v,
						"color" : identify_color,
						"segments" : [{
							"start" : 0,
							"end" : value_1[i],
							"colors" : value_1_color,
							"task": "20Day"
						},{
							"start" : 0,
							"end" : value_3[i],
							"colors" : value_3_color,
							"task": "1Day"
						},{
							"start" : 0,	
							"end" : value_2[i],
							"colors" : value_2_color,
							"task": "3Day"
						}]
					});
				}

				else if(check[0] == 1 && check[1] == 0 && check[2] == 2 )
				{
					if(i_color !== '')
					{
						identify_color = i_color;
					}

					data.push({
						"category" : v,
						"color" : identify_color,
						"segments" : [{
							"start" : 0,
							"end" : value_2[i],
							"colors" : value_2_color,
							"task": "3Day"
						},{
							"start" : 0,
							"end" : value_1[i],
							"colors" : value_1_color,
							"task": "20Day"
						},{
							"start" : 0,	
							"end" : value_3[i],
							"colors" : value_3_color,
							"task": "1Day"
						}]
					});
				}

				else if(check[0] == 1 && check[1] == 2 && check[2] == 0 )
				{
					if(i_color !== '')
					{
						identify_color = i_color;
					}

					data.push({
						"category" : v,
						"color" : identify_color,
						"segments" : [{
							"start" : 0,
							"end" : value_2[i],
							"colors" : value_2_color,
							"task": "3Day"
						},{
							"start" : 0,
							"end" : value_3[i],
							"colors" : value_3_color,
							"task": "1Day"
						},{
							"start" : 0,	
							"end" : value_1[i],
							"colors" : value_1_color,
							"task": "20Day"
						}]
					});
				}

				else if(check[0] == 2 && check[1] == 0 && check[2] == 1 )
				{
					if(i_color !== '')
					{
						identify_color = i_color;
					}

					data.push({
						"category" : v,
						"color" : identify_color,
						"segments" : [{
							"start" : 0,
							"end" : value_3[i],
							"colors" : value_3_color,
							"task": "1Day"
						},{
							"start" : 0,
							"end" : value_1[i],
							"colors" : value_1_color,
							"task": "20Day"
						},{
							"start" : 0,	
							"end" : value_2[i],
							"colors" : value_2_color,
							"task": "3Day"
						}]
					});
				}

				else if(check[0] == 2 && check[1] == 1 && check[2] == 0 )
				{
					if(i_color !== '')
					{
						identify_color = i_color;
					}

					data.push({
						"category" : v,
						"color" : identify_color,
						"segments" : [{
							"start" : 0,
							"end" : value_3[i],
							"colors" : value_3_color,
							"task": "1Day"
						},{
							"start" : 0,
							"end" : value_2[i],
							"colors" : value_2_color,
							"task": "3Day"
						},{
							"start" : 0,	
							"end" : value_1[i],
							"colors" : value_1_color,
							"task": "20Day"
						}]
					});
				}

                i = i + 1;
            });
			
			// console.log(test);

            var d = new Date();
            rank_startdate = d.getDate()+"-"+(d.getMonth()+1)+"-"+d.getFullYear();
            rank_enddate = d.getDate()+"-"+(d.getMonth()+1)+"-"+d.getFullYear();;


            AmCharts.addInitHandler( function ( chart ) {


                // set base values
                var categoryWidth = 25;
                  
                // calculate bottom margin based on number of data points
                var chartHeight = categoryWidth * chart.dataProvider.length;
                
                // set the value
                chart.div.style.height = chartHeight + 'px';
        
            }, ['gantt'] );
            

            

            var chart = AmCharts.makeChart( "ranking_chart", {
				"type": "gantt",
				"theme": "none",
				"titles": [{
					"text": rank_title,
					"size": 16
				}],
				"marginRight": 70,
				"columnWidth": 1,
				"valueAxis": {
					"type": "number"
				},
				"brightnessStep": 10,
				"graph": {
					"fillAlphas": 4,
					"balloonText": "Series '[[task]] Lookback Point [[v]]<br>Value:[[value]]",
          			"pointerWidth": 50,
				},
				"rotate": true,
				"addClassNames": true,
				"categoryField": "category",
				"segmentsField": "segments",
				"colorField": "colors",
				"startDate": "2015-01-01",
				"startField": "start",
				"endField": "end",
				"durationField": "duration",
			    "listeners": [{
					"event": "rendered",
					"method": updateLabels
					},{
				    "event": "resized",
				    "method": updateLabels
			    }],
				"dataProvider": data,
				"chartCursor": {
					"cursorColor":"#55bb76",
					"valueBalloonsEnabled": false,
					"cursorAlpha": 0,
					"valueLineAlpha":0.5,
					"valueLineBalloonEnabled": true,
					"valueLineEnabled": true,
					"zoomable":false,
					"valueZoomable":true
				},  
				"export": {
					"enabled": true
				}
			} );

			function updateLabels(event) { 
	          	var labels = event.chart.chartDiv.getElementsByClassName("amcharts-axis-label");
	          	// console.log(data.length);

	          	for (var i = 0; i < (data.length); i++) {
	            	var color = event.chart.dataProvider[i].color; 
	            	// console.log(labels[i]);
	            	// console.log(color);
			            if (color !== undefined)  { 
			              	labels[i].setAttribute("fill", color);
			            }
	          	}
	        }


            /* ------------------
            ---------------------
            Ranking Chart Done
            ---------------------
            --------------------- */


        }
    });

}




$(document).on('click', '.chart-pane-tab', function(event) {

    var board_value = $('.chart-title-text').html();
    board_value=board_value.split(":");
    board_value=board_value[1].replace(/\s+/g, '');

    var tab_value=$(this).text();
    tab_value = tab_value.replace(/\s+/g, '');

    var anti_val = '';

    if(tab_value==tab_value_index[2] + "K" || tab_value==tab_value_index[1] + "K"  ||  tab_value==tab_value_index[0] + "K")
    {
    	$(".chart_loader").show();
    }

    $.ajax
    ({
        url: '/getchartdata',
        type: 'get',
        dataType: 'json', // added data type
        success: function(result)
        {
          $(".chart_loader").hide();

          var date = [];
          var system = [];
          var anti_system = [];
          var benchmark = [];
          var cumper = [];

          var anti_system_cum = [];
          var benchmark_cum = [];

          var system_pnl = [];
          var system_pnl_percent = [];

          var chart_title = '';
          var anti_system_cum_title = '';
          var system_pnl_percent_title = '';
          btn_click = 3;

          if(tab_value==tab_value_index[2] + "K")
          {

            chart_title = "v4micro 20Days Historical Performance: "+board_value+", Anti-"+board_value+", Benchmark";
            $.each(result.v4micro_performance, function(l,m) {
              if(l==board_value)
              {
                  $.each(this, function(k, v) {

                      date.push(k);
                      system.push(v);

                  });
              }
              if(board_value=='AntiHighestEquity' && l=='HighestEquity')
              {
                  anti_val = 'HighestEquity';
                  $.each(this, function(k, v) {

                            anti_system.push(v);

                        });
                  anti_system_cum_title = l +'_cumper';
                  system_pnl_percent_title = l + '_pnl';

                  anti_system_cum_val = l +' Cum %';
                  system_pnl_percent_val = l + " Daily P&L %";

                  chart_title = "v4futures 20Days Historical Performance: "+board_value+", " + l + ", Benchmark";
              }
              else if(l=='AntiHighestEquity' && board_value=='HighestEquity')
              {
                  anti_val = 'AntiHighestEquity';
                 $.each(this, function(k, v) {

                            anti_system.push(v);

                        });
                 anti_system_cum_title = l +'_cumper';
                 system_pnl_percent_title = l + '_pnl';

                 anti_system_cum_val = l +' Cum %';
                 system_pnl_percent_val = l + " Daily P&L %";

                 chart_title = "v4futures 20Days Historical Performance: "+board_value+", " + l + ", Benchmark";
              }



              else if(board_value=='Previous' && l=='Anti-Previous')
              {
                  anti_val = 'Anti-Previous';
                  $.each(this, function(k, v) {

                            anti_system.push(v);

                        });
                  anti_system_cum_title = l +'_cumper';
                  system_pnl_percent_title = l + '_pnl';

                  anti_system_cum_val = l +' Cum %';
                  system_pnl_percent_val = l + " Daily P&L %";

                  chart_title = "v4futures 20Days Historical Performance: "+board_value+", " + l + ", Benchmark";
              }
              else if(l=='Previous' && board_value=='Anti-Previous')
              {
                  anti_val = 'Previous';
                  $.each(this, function(k, v) {

                            anti_system.push(v);

                        });
                  anti_system_cum_title = l +'_cumper';
                  system_pnl_percent_title = l + '_pnl';

                  anti_system_cum_val = l +' Cum %';
                  system_pnl_percent_val = l + " Daily P&L %";

                  chart_title = "v4futures 20Days Historical Performance: "+board_value+", " + l + ", Benchmark";
              }



              else if(board_value=='RiskOn' && l=='RiskOff')
              {
                  anti_val = 'RiskOff';
                  $.each(this, function(k, v) {

                            anti_system.push(v);

                        });
                  anti_system_cum_title = l +'_cumper';
                  system_pnl_percent_title = l + '_pnl';

                  anti_system_cum_val = l +' Cum %';
                  system_pnl_percent_val = l + " Daily P&L %";

                  chart_title = "v4futures 20Days Historical Performance: "+board_value+", " + l + ", Benchmark";
              }
              else if(l=='RiskOn' && board_value=='RiskOff')
              {
                  anti_val = 'RiskOn';
                  $.each(this, function(k, v) {

                            anti_system.push(v);

                        });
                  anti_system_cum_title = l +'_cumper';
                  system_pnl_percent_title = l + '_pnl';

                  anti_system_cum_val = l +' Cum %';
                  system_pnl_percent_val = l + " Daily P&L %";

                  chart_title = "v4futures 20Days Historical Performance: "+board_value+", " + l + ", Benchmark";
              }



              else if(board_value=='50/50' && l=='Anti50/50')
              {
                  anti_val = 'Anti50/50';
                  $.each(this, function(k, v) {

                            anti_system.push(v);

                        });
                  anti_system_cum_title = l +'_cumper';
                  system_pnl_percent_title = l + '_pnl';

                  anti_system_cum_val = l +' Cum %';
                  system_pnl_percent_val = l + " Daily P&L %";

                  chart_title = "v4futures 20Days Historical Performance: "+board_value+", " + l + ", Benchmark";
              }
              else if(l=='50/50' && board_value=='Anti50/50')
              { 
                  anti_val = '50/50';
                  $.each(this, function(k, v) {

                            anti_system.push(v);

                        });
                  anti_system_cum_title = l +'_cumper';
                  system_pnl_percent_title = l + '_pnl';

                  anti_system_cum_val = l +' Cum %';
                  system_pnl_percent_val = l + " Daily P&L %";

                  chart_title = "v4futures 20Days Historical Performance: "+board_value+", " + l + ", Benchmark";
              }



              else if(board_value=='LowestEquity' && l=='AntiLowestEquity')
              {
                  anti_val = 'AntiLowestEquity';
                  $.each(this, function(k, v) {

                            anti_system.push(v);

                        });
                  anti_system_cum_title = l +'_cumper';
                  system_pnl_percent_title = l + '_pnl';

                  anti_system_cum_val = l +' Cum %';
                  system_pnl_percent_val = l + " Daily P&L %";

                  chart_title = "v4futures 20Days Historical Performance: "+board_value+", " + l + ", Benchmark";
              }
              else if(l=='LowestEquity' && board_value=='AntiLowestEquity')
              {
                  anti_val = 'LowestEquity';
                  $.each(this, function(k, v) {

                            anti_system.push(v);

                        });
                  anti_system_cum_title = l +'_cumper';
                  system_pnl_percent_title = l + '_pnl';

                  anti_system_cum_val = l +' Cum %';
                  system_pnl_percent_val = l + " Daily P&L %";

                  chart_title = "v4futures 20Days Historical Performance: "+board_value+", " + l + ", Benchmark";
              }

              else if(board_value=='Custom' && l=='Anti-Custom')
              {
                  anti_val = 'Anti-Custom';
                  $.each(this, function(k, v) {

                            anti_system.push(v);

                        });
                  anti_system_cum_title = l +'_cumper';
                  system_pnl_percent_title = l + '_pnl';

                  anti_system_cum_val = l +' Cum %';
                  system_pnl_percent_val = l + " Daily P&L %";

                  chart_title = "v4futures 20Days Historical Performance: "+board_value+", " + l + ", Benchmark";
              }
              else if(l=='Custom' && board_value=='Anti-Custom')
              {
                  anti_val = 'Custom';
                  $.each(this, function(k, v) {

                            anti_system.push(v);

                        });
                  anti_system_cum_title = l +'_cumper';
                  system_pnl_percent_title = l + '_pnl';

                  anti_system_cum_val = l +' Cum %';
                  system_pnl_percent_val = l + " Daily P&L %";

                  chart_title = "v4futures 20Days Historical Performance: "+board_value+", " + l + ", Benchmark";
              }


              else if(board_value=='Seasonality' && l=='Anti-Seasonality')
              {
                  anti_val = 'Anti-Seasonality';
                  $.each(this, function(k, v) {

                            anti_system.push(v);

                        });
                  anti_system_cum_title = l +'_cumper';
                  system_pnl_percent_title = l + '_pnl';

                  anti_system_cum_val = l +' Cum %';
                  system_pnl_percent_val = l + " Daily P&L %";

                  chart_title = "v4futures 20Days Historical Performance: "+board_value+", " + l + ", Benchmark";
              }
              else if(l=='Seasonality' && board_value=='Anti-Seasonality')
              {
                  anti_val = 'Seasonality';
                  $.each(this, function(k, v) {

                            anti_system.push(v);

                        });
                  anti_system_cum_title = l +'_cumper';
                  system_pnl_percent_title = l + '_pnl';

                  anti_system_cum_val = l +' Cum %';
                  system_pnl_percent_val = l + " Daily P&L %";

                  chart_title = "v4futures 20Days Historical Performance: "+board_value+", " + l + ", Benchmark";
              }
              else
              {
                 if(l=='Anti-'+board_value)
                  {
                      anti_val = 'Anti-'+ board_value;
                      $.each(this, function(k, v) {

                          anti_system.push(v);

                      });
                      anti_system_cum_title = 'Anti-'+board_value+'_cumper';
                      system_pnl_percent_title = "Anti-" + board_value + "_pnl";

                      anti_system_cum_val = 'Anti-'+ board_value+' Cum %';
                      system_pnl_percent_val = "Anti-" + board_value + " Daily P&L %";
                  }
              }

              if(l==anti_system_cum_title)
              {
                  $.each(this, function(k, v) {

                      v = v + " %";
                      anti_system_cum.push(v);

                  });
              }
              
              if(l=='benchmark')
              {
                  $.each(this, function(k, v) {

                      benchmark.push(v);

                  });
              }
              if(l=='benchmark_cumper')
              {
                  $.each(this, function(k, v) {

                      v = v + " %";
                      benchmark_cum.push(v);

                  });
              }
              if(l==board_value+"_cumper")
              {
                  $.each(this, function(k, v) {

                      cumper_val = board_value + " Cum %";
                      v = v + " %";
                      cumper.push(v);

                  });
              }
              if(l == board_value + "_pnl")
              {
                  $.each(this, function(k, v) {

                      system_pnl_val = board_value + " Daily P&L %";
                      v = v * 100;
                      v = parseFloat(v).toFixed(2);
                      v = v + " %";
                      system_pnl.push(v);

                  });
              }
              if(l == system_pnl_percent_title )
              {
                  $.each(this, function(k, v) {

                      v = v * 100;
                      v = parseFloat(v).toFixed(2);
                      v = v + " %";
                      system_pnl_percent.push(v);

                  });
              }
            });
          }
          else if(tab_value==tab_value_index[1] + "K")
          {
            chart_title = "v4mini 20Days Historical Performance: "+board_value+", Anti-"+board_value+", Benchmark";
            $.each(result.v4mini_performance, function(l,m) {
              if(l==board_value)
              {
                  $.each(this, function(k, v) {

                      date.push(k);
                      system.push(v);

                  });
              }
              if(board_value=='AntiHighestEquity' && l=='HighestEquity')
              {
                  anti_val = 'HighestEquity';
                  $.each(this, function(k, v) {

                            anti_system.push(v);

                        });
                  anti_system_cum_title = l +'_cumper';
                  system_pnl_percent_title = l + '_pnl';

                  anti_system_cum_val = l +' Cum %';
                  system_pnl_percent_val = l + " Daily P&L %";

                  chart_title = "v4futures 20Days Historical Performance: "+board_value+", " + l + ", Benchmark";
              }
              else if(l=='AntiHighestEquity' && board_value=='HighestEquity')
              {
                  anti_val = 'AntiHighestEquity';
                 $.each(this, function(k, v) {

                            anti_system.push(v);

                        });
                 anti_system_cum_title = l +'_cumper';
                 system_pnl_percent_title = l + '_pnl';

                 anti_system_cum_val = l +' Cum %';
                 system_pnl_percent_val = l + " Daily P&L %";

                 chart_title = "v4futures 20Days Historical Performance: "+board_value+", " + l + ", Benchmark";
              }



              else if(board_value=='Previous' && l=='Anti-Previous')
              {
                  anti_val = 'Anti-Previous';
                  $.each(this, function(k, v) {

                            anti_system.push(v);

                        });
                  anti_system_cum_title = l +'_cumper';
                  system_pnl_percent_title = l + '_pnl';

                  anti_system_cum_val = l +' Cum %';
                  system_pnl_percent_val = l + " Daily P&L %";

                  chart_title = "v4futures 20Days Historical Performance: "+board_value+", " + l + ", Benchmark";
              }
              else if(l=='Previous' && board_value=='Anti-Previous')
              {
                  anti_val = 'Previous';
                  $.each(this, function(k, v) {

                            anti_system.push(v);

                        });
                  anti_system_cum_title = l +'_cumper';
                  system_pnl_percent_title = l + '_pnl';

                  anti_system_cum_val = l +' Cum %';
                  system_pnl_percent_val = l + " Daily P&L %";

                  chart_title = "v4futures 20Days Historical Performance: "+board_value+", " + l + ", Benchmark";
              }



              else if(board_value=='RiskOn' && l=='RiskOff')
              {
                  anti_val = 'RiskOff';
                  $.each(this, function(k, v) {

                            anti_system.push(v);

                        });
                  anti_system_cum_title = l +'_cumper';
                  system_pnl_percent_title = l + '_pnl';

                  anti_system_cum_val = l +' Cum %';
                  system_pnl_percent_val = l + " Daily P&L %";

                  chart_title = "v4futures 20Days Historical Performance: "+board_value+", " + l + ", Benchmark";
              }
              else if(l=='RiskOn' && board_value=='RiskOff')
              {
                  anti_val = 'RiskOn';
                  $.each(this, function(k, v) {

                            anti_system.push(v);

                        });
                  anti_system_cum_title = l +'_cumper';
                  system_pnl_percent_title = l + '_pnl';

                  anti_system_cum_val = l +' Cum %';
                  system_pnl_percent_val = l + " Daily P&L %";

                  chart_title = "v4futures 20Days Historical Performance: "+board_value+", " + l + ", Benchmark";
              }



              else if(board_value=='50/50' && l=='Anti50/50')
              {
                  anti_val = 'Anti50/50';
                  $.each(this, function(k, v) {

                            anti_system.push(v);

                        });
                  anti_system_cum_title = l +'_cumper';
                  system_pnl_percent_title = l + '_pnl';

                  anti_system_cum_val = l +' Cum %';
                  system_pnl_percent_val = l + " Daily P&L %";

                  chart_title = "v4futures 20Days Historical Performance: "+board_value+", " + l + ", Benchmark";
              }
              else if(l=='50/50' && board_value=='Anti50/50')
              { 
                  anti_val = '50/50';
                  $.each(this, function(k, v) {

                            anti_system.push(v);

                        });
                  anti_system_cum_title = l +'_cumper';
                  system_pnl_percent_title = l + '_pnl';

                  anti_system_cum_val = l +' Cum %';
                  system_pnl_percent_val = l + " Daily P&L %";

                  chart_title = "v4futures 20Days Historical Performance: "+board_value+", " + l + ", Benchmark";
              }



              else if(board_value=='LowestEquity' && l=='AntiLowestEquity')
              {
                  anti_val = 'AntiLowestEquity';
                  $.each(this, function(k, v) {

                            anti_system.push(v);

                        });
                  anti_system_cum_title = l +'_cumper';
                  system_pnl_percent_title = l + '_pnl';

                  anti_system_cum_val = l +' Cum %';
                  system_pnl_percent_val = l + " Daily P&L %";

                  chart_title = "v4futures 20Days Historical Performance: "+board_value+", " + l + ", Benchmark";
              }
              else if(l=='LowestEquity' && board_value=='AntiLowestEquity')
              {
                  anti_val = 'LowestEquity';
                  $.each(this, function(k, v) {

                            anti_system.push(v);

                        });
                  anti_system_cum_title = l +'_cumper';
                  system_pnl_percent_title = l + '_pnl';

                  anti_system_cum_val = l +' Cum %';
                  system_pnl_percent_val = l + " Daily P&L %";

                  chart_title = "v4futures 20Days Historical Performance: "+board_value+", " + l + ", Benchmark";
              }

              else if(board_value=='Custom' && l=='Anti-Custom')
              {
                  anti_val = 'Anti-Custom';
                  $.each(this, function(k, v) {

                            anti_system.push(v);

                        });
                  anti_system_cum_title = l +'_cumper';
                  system_pnl_percent_title = l + '_pnl';

                  anti_system_cum_val = l +' Cum %';
                  system_pnl_percent_val = l + " Daily P&L %";

                  chart_title = "v4futures 20Days Historical Performance: "+board_value+", " + l + ", Benchmark";
              }
              else if(l=='Custom' && board_value=='Anti-Custom')
              {
                  anti_val = 'Custom';
                  $.each(this, function(k, v) {

                            anti_system.push(v);

                        });
                  anti_system_cum_title = l +'_cumper';
                  system_pnl_percent_title = l + '_pnl';

                  anti_system_cum_val = l +' Cum %';
                  system_pnl_percent_val = l + " Daily P&L %";

                  chart_title = "v4futures 20Days Historical Performance: "+board_value+", " + l + ", Benchmark";
              }


              else if(board_value=='Seasonality' && l=='Anti-Seasonality')
              {
                  anti_val = 'Anti-Seasonality';
                  $.each(this, function(k, v) {

                            anti_system.push(v);

                        });
                  anti_system_cum_title = l +'_cumper';
                  system_pnl_percent_title = l + '_pnl';

                  anti_system_cum_val = l +' Cum %';
                  system_pnl_percent_val = l + " Daily P&L %";

                  chart_title = "v4futures 20Days Historical Performance: "+board_value+", " + l + ", Benchmark";
              }
              else if(l=='Seasonality' && board_value=='Anti-Seasonality')
              {
                  anti_val = 'Seasonality';
                  $.each(this, function(k, v) {

                            anti_system.push(v);

                        });
                  anti_system_cum_title = l +'_cumper';
                  system_pnl_percent_title = l + '_pnl';

                  anti_system_cum_val = l +' Cum %';
                  system_pnl_percent_val = l + " Daily P&L %";

                  chart_title = "v4futures 20Days Historical Performance: "+board_value+", " + l + ", Benchmark";
              }
              else
              {
                 if(l=='Anti-'+board_value)
                  {
                      anti_val = 'Anti-'+ board_value;
                      $.each(this, function(k, v) {

                          anti_system.push(v);

                      });
                      anti_system_cum_title = 'Anti-'+board_value+'_cumper';
                      system_pnl_percent_title = "Anti-" + board_value + "_pnl";

                      anti_system_cum_val = 'Anti-'+ board_value+' Cum %';
                      system_pnl_percent_val = "Anti-" + board_value + " Daily P&L %";
                  }
              }

              if(l==anti_system_cum_title)
              {
                  $.each(this, function(k, v) {

                      v = v + " %";
                      anti_system_cum.push(v);

                  });
              }
              
              if(l=='benchmark')
              {
                  $.each(this, function(k, v) {

                      benchmark.push(v);

                  });
              }
              if(l=='benchmark_cumper')
              {
                  $.each(this, function(k, v) {

                      v = v + " %";
                      benchmark_cum.push(v);

                  });
              }
              if(l==board_value+"_cumper")
              {
                  $.each(this, function(k, v) {

                      cumper_val = board_value + " Cum %";
                      v = v + " %";
                      cumper.push(v);

                  });
              }
              if(l == board_value + "_pnl")
              {
                  $.each(this, function(k, v) {

                      system_pnl_val = board_value + " Daily P&L %";
                      v = v * 100;
                      v = parseFloat(v).toFixed(2);
                      v = v + " %";
                      system_pnl.push(v);

                  });
              }
              if(l == system_pnl_percent_title )
              {
                  $.each(this, function(k, v) {

                      v = v * 100;
                      v = parseFloat(v).toFixed(2);
                      v = v + " %";
                      system_pnl_percent.push(v);

                  });
              }
            });
          }
           else if(tab_value==tab_value_index[0] + "K")
          {
            chart_title = "v4futures 20Days Historical Performance: "+board_value+", Anti-"+board_value+", Benchmark";
            $.each(result.v4futures_performance, function(l,m) {
              if(l==board_value)
              {
                  $.each(this, function(k, v) {

                      date.push(k);
                      system.push(v);

                  });
              }
              if(board_value=='AntiHighestEquity' && l=='HighestEquity')
              {
                  anti_val = 'HighestEquity';
                  $.each(this, function(k, v) {

                            anti_system.push(v);

                        });
                  anti_system_cum_title = l +'_cumper';
                  system_pnl_percent_title = l + '_pnl';

                  anti_system_cum_val = l +' Cum %';
                  system_pnl_percent_val = l + " Daily P&L %";

                  chart_title = "v4futures 20Days Historical Performance: "+board_value+", " + l + ", Benchmark";
              }
              else if(l=='AntiHighestEquity' && board_value=='HighestEquity')
              {
                  anti_val = 'AntiHighestEquity';
                 $.each(this, function(k, v) {

                            anti_system.push(v);

                        });
                 anti_system_cum_title = l +'_cumper';
                 system_pnl_percent_title = l + '_pnl';

                 anti_system_cum_val = l +' Cum %';
                 system_pnl_percent_val = l + " Daily P&L %";

                 chart_title = "v4futures 20Days Historical Performance: "+board_value+", " + l + ", Benchmark";
              }



              else if(board_value=='Previous' && l=='Anti-Previous')
              {
                  anti_val = 'Anti-Previous';
                  $.each(this, function(k, v) {

                            anti_system.push(v);

                        });
                  anti_system_cum_title = l +'_cumper';
                  system_pnl_percent_title = l + '_pnl';

                  anti_system_cum_val = l +' Cum %';
                  system_pnl_percent_val = l + " Daily P&L %";

                  chart_title = "v4futures 20Days Historical Performance: "+board_value+", " + l + ", Benchmark";
              }
              else if(l=='Previous' && board_value=='Anti-Previous')
              {
                  anti_val = 'Previous';
                  $.each(this, function(k, v) {

                            anti_system.push(v);

                        });
                  anti_system_cum_title = l +'_cumper';
                  system_pnl_percent_title = l + '_pnl';

                  anti_system_cum_val = l +' Cum %';
                  system_pnl_percent_val = l + " Daily P&L %";

                  chart_title = "v4futures 20Days Historical Performance: "+board_value+", " + l + ", Benchmark";
              }



              else if(board_value=='RiskOn' && l=='RiskOff')
              {
                  anti_val = 'RiskOff';
                  $.each(this, function(k, v) {

                            anti_system.push(v);

                        });
                  anti_system_cum_title = l +'_cumper';
                  system_pnl_percent_title = l + '_pnl';

                  anti_system_cum_val = l +' Cum %';
                  system_pnl_percent_val = l + " Daily P&L %";

                  chart_title = "v4futures 20Days Historical Performance: "+board_value+", " + l + ", Benchmark";
              }
              else if(l=='RiskOn' && board_value=='RiskOff')
              {
                  anti_val = 'RiskOn';
                  $.each(this, function(k, v) {

                            anti_system.push(v);

                        });
                  anti_system_cum_title = l +'_cumper';
                  system_pnl_percent_title = l + '_pnl';

                  anti_system_cum_val = l +' Cum %';
                  system_pnl_percent_val = l + " Daily P&L %";

                  chart_title = "v4futures 20Days Historical Performance: "+board_value+", " + l + ", Benchmark";
              }



              else if(board_value=='50/50' && l=='Anti50/50')
              {
                  anti_val = 'Anti50/50';
                  $.each(this, function(k, v) {

                            anti_system.push(v);

                        });
                  anti_system_cum_title = l +'_cumper';
                  system_pnl_percent_title = l + '_pnl';

                  anti_system_cum_val = l +' Cum %';
                  system_pnl_percent_val = l + " Daily P&L %";

                  chart_title = "v4futures 20Days Historical Performance: "+board_value+", " + l + ", Benchmark";
              }
              else if(l=='50/50' && board_value=='Anti50/50')
              { 
                  anti_val = '50/50';
                  $.each(this, function(k, v) {

                            anti_system.push(v);

                        });
                  anti_system_cum_title = l +'_cumper';
                  system_pnl_percent_title = l + '_pnl';

                  anti_system_cum_val = l +' Cum %';
                  system_pnl_percent_val = l + " Daily P&L %";

                  chart_title = "v4futures 20Days Historical Performance: "+board_value+", " + l + ", Benchmark";
              }



              else if(board_value=='LowestEquity' && l=='AntiLowestEquity')
              {
                  anti_val = 'AntiLowestEquity';
                  $.each(this, function(k, v) {

                            anti_system.push(v);

                        });
                  anti_system_cum_title = l +'_cumper';
                  system_pnl_percent_title = l + '_pnl';

                  anti_system_cum_val = l +' Cum %';
                  system_pnl_percent_val = l + " Daily P&L %";

                  chart_title = "v4futures 20Days Historical Performance: "+board_value+", " + l + ", Benchmark";
              }
              else if(l=='LowestEquity' && board_value=='AntiLowestEquity')
              {
                  anti_val = 'LowestEquity';
                  $.each(this, function(k, v) {

                            anti_system.push(v);

                        });
                  anti_system_cum_title = l +'_cumper';
                  system_pnl_percent_title = l + '_pnl';

                  anti_system_cum_val = l +' Cum %';
                  system_pnl_percent_val = l + " Daily P&L %";

                  chart_title = "v4futures 20Days Historical Performance: "+board_value+", " + l + ", Benchmark";
              }

              else if(board_value=='Custom' && l=='Anti-Custom')
              {
                  anti_val = 'Anti-Custom';
                  $.each(this, function(k, v) {

                            anti_system.push(v);

                        });
                  anti_system_cum_title = l +'_cumper';
                  system_pnl_percent_title = l + '_pnl';

                  anti_system_cum_val = l +' Cum %';
                  system_pnl_percent_val = l + " Daily P&L %";

                  chart_title = "v4futures 20Days Historical Performance: "+board_value+", " + l + ", Benchmark";
              }
              else if(l=='Custom' && board_value=='Anti-Custom')
              {
                  anti_val = 'Custom';
                  $.each(this, function(k, v) {

                            anti_system.push(v);

                        });
                  anti_system_cum_title = l +'_cumper';
                  system_pnl_percent_title = l + '_pnl';

                  anti_system_cum_val = l +' Cum %';
                  system_pnl_percent_val = l + " Daily P&L %";

                  chart_title = "v4futures 20Days Historical Performance: "+board_value+", " + l + ", Benchmark";
              }


              else if(board_value=='Seasonality' && l=='Anti-Seasonality')
              {
                  anti_val = 'Anti-Seasonality';
                  $.each(this, function(k, v) {

                            anti_system.push(v);

                        });
                  anti_system_cum_title = l +'_cumper';
                  system_pnl_percent_title = l + '_pnl';

                  anti_system_cum_val = l +' Cum %';
                  system_pnl_percent_val = l + " Daily P&L %";

                  chart_title = "v4futures 20Days Historical Performance: "+board_value+", " + l + ", Benchmark";
              }
              else if(l=='Seasonality' && board_value=='Anti-Seasonality')
              {
                  anti_val = 'Seasonality';
                  $.each(this, function(k, v) {

                            anti_system.push(v);

                        });
                  anti_system_cum_title = l +'_cumper';
                  system_pnl_percent_title = l + '_pnl';

                  anti_system_cum_val = l +' Cum %';
                  system_pnl_percent_val = l + " Daily P&L %";

                  chart_title = "v4futures 20Days Historical Performance: "+board_value+", " + l + ", Benchmark";
              }
              else
              {
                 if(l=='Anti-'+board_value)
                  {
                      anti_val = 'Anti-'+ board_value;
                      $.each(this, function(k, v) {

                          anti_system.push(v);

                      });
                      anti_system_cum_title = 'Anti-'+board_value+'_cumper';
                      system_pnl_percent_title = "Anti-" + board_value + "_pnl";

                      anti_system_cum_val = 'Anti-'+ board_value+' Cum %';
                      system_pnl_percent_val = "Anti-" + board_value + " Daily P&L %";
                  }
              }

              if(l==anti_system_cum_title)
              {
                  $.each(this, function(k, v) {

                      v = v + " %";
                      anti_system_cum.push(v);

                  });
              }
              
              if(l=='benchmark')
              {
                  $.each(this, function(k, v) {

                      benchmark.push(v);

                  });
              }
              if(l=='benchmark_cumper')
              {
                  $.each(this, function(k, v) {

                      v = v + " %";
                      benchmark_cum.push(v);

                  });
              }
              if(l==board_value+"_cumper")
              {
                  $.each(this, function(k, v) {

                      cumper_val = board_value + " Cum %";
                      v = v + " %";
                      cumper.push(v);

                  });
              }
              if(l == board_value + "_pnl")
              {
                  $.each(this, function(k, v) {

                      system_pnl_val = board_value + " Daily P&L %";
                      v = v * 100;
                      v = parseFloat(v).toFixed(2);
                      v = v + " %";
                      system_pnl.push(v);

                  });
              }
              if(l == system_pnl_percent_title )
              {
                  $.each(this, function(k, v) {

                      v = v * 100;
                      v = parseFloat(v).toFixed(2);
                      v = v + " %";
                      system_pnl_percent.push(v);

                  });
              }

            });
          }
          else
          {
            console.log("NO BOARD");
            // ruturn false;
          }

          var max_system = Math.max.apply(Math,system);
          var min_system = Math.min.apply(Math,system);

          var max_anti_system = Math.max.apply(Math,anti_system);
          var min_anti_system = Math.min.apply(Math,anti_system);

          var max_benchmark = Math.max.apply(Math,benchmark);
          var min_benchmark = Math.min.apply(Math,benchmark);

          var max = Math.max(max_system, max_anti_system, max_benchmark);
          var min = Math.min(min_system, min_anti_system, min_benchmark);
          
          max = Math.round(max) + 5000;
    	  min = Math.round(min) - 5000;

          var tot='';
          tot+=min.toString();
          tot+=":";
          tot+=max.toString();
          tot+=":1000";

          // console.log(anti_system);
          // console.log(system_pnl_percent);
          // console.log(anti_system_cum);
      
          var myConfig =  {
            "type":"line",
            "utc": true,
            "title": {
              "text": chart_title,
              "font-size": "14px",
              "adjust-layout": true
            },
            "plotarea": {
              "margin": "dynamic 45 60 dynamic",
            },
            "scale-x":{
              "values":date,
              "transform": {
                "type": "date",
                "all": "%d %M %Y",
                "guide": {
                  "visible": false
                },
              },
              "item":{  
                "font-angle":315,  
              } 
            },
            "scale-y":{
              "values":tot,
              "guide": {
                "line-style": "dashed"
              },
              "thousands-separator":",",
            },
            "plot": {
              "highlight": true,
              "tooltip-text": "%t: %v<br>Date:%k",
              "shadow": 0,
              "line-width": "2px",
              "marker": {
                "type": "circle",
                "size": 1
              },
              "highlight-state": {
                "line-width": 3
              },
              "animation": {
                "effect": 1,
                "sequence": 2,
                "speed": 100,
              },
            },
            "crosshair-x": {
              "line-color": "#efefef",
              "plot-label": {
                "border-radius": "5px",
                "border-width": "1px",
                "border-color": "#f6f7f8",
                "padding": "10px",
                "font-weight": "bold",
                "thousands-separator":",",
                "font-size": "14px",
              },
              "scale-label": {
                "font-color": "#000",
                "background-color": "#f6f7f8",
                "border-radius": "5px"
              },
            },
            "tooltip": {
              "visible": false
            },
            "series":[
              {"values":system,
              "line-color":"#0000FF",
              "line-style":"line",
              "text": board_value,
              "legend-item": {
                "background-color": "#007790",
                "borderRadius": "5",
                "font-color": "white"
                },
                "marker": {
                  "background-color": "#da534d",
                  "border-width": 0,
                  "shadow": 0,
                  "border-color": "#faa39f"
                },
                "highlight-marker": {
                  "size": 6,
                  "background-color": "#da534d",
                }
              },
              {"values":system_pnl,
              "line-color":"#0000FF",
              "line-style":"line",
              "text": system_pnl_val,
              "legend-item": {
                "background-color": "#007790",
                "borderRadius": "5",
                "font-color": "white"
                },
                "marker": {
                  "background-color": "#da534d",
                  "border-width": 0,
                  "shadow": 0 ,
                  "border-color": "#faa39f"
                },
                "highlight-marker": {
                  "size": 6,
                  "background-color": "#da534d",
                }
              },
              {"values":cumper,
              "line-color":"#0000FF",
              "line-style":"line",
              "text": cumper_val,
              "legend-item": {
                "background-color": "#007790",
                "borderRadius": "5",
                "font-color": "white"
                },
                "marker": {
                  "background-color": "#da534d",
                  "border-width": 0,
                  "shadow": 0 ,
                  "border-color": "#faa39f"
                },
                "highlight-marker": {
                  "size": 6,
                  "background-color": "#da534d",
                }
              },
              {"values":anti_system,
              "line-color":"#0B850C",
              "line-style":"line",
              "text": anti_val,
              "legend-item": {
                "background-color": "#007790",
                "borderRadius": "5",
                "font-color": "white"
                },
                "marker": {
                  "background-color": "#da534d",
                  "border-width": 0,
                  "shadow": 0 ,
                  "border-color": "#faa39f"
                },
                "highlight-marker": {
                  "size": 6,
                  "background-color": "#da534d",
                }
              },
              {"values":system_pnl_percent,
              "line-color":"#0B850C",
              "line-style":"line",
              "text": system_pnl_percent_val,
              "legend-item": {
                "background-color": "#007790",
                "borderRadius": "5",
                "font-color": "white"
                },
                "marker": {
                  "background-color": "#da534d",
                  "border-width": 0,
                  "shadow": 0 ,
                  "border-color": "#faa39f"
                },
                "highlight-marker": {
                  "size": 6,
                  "background-color": "#da534d",
                }
              },
              {"values":anti_system_cum,
              "line-color":"#0B850C",
              "line-style":"line",
              "text": anti_system_cum_val,
              "legend-item": {
                "background-color": "#007790",
                "borderRadius": "5",
                "font-color": "white"
                },
                "marker": {
                  "background-color": "#da534d",
                  "border-width": 0,
                  "shadow": 0 ,
                  "border-color": "#faa39f"
                },
                "highlight-marker": {
                  "size": 6,
                  "background-color": "#da534d",
                }
              },
              {"values":benchmark,
              "line-color":"#ff0000",
              "line-style":"line",
              "text": "Benchmark",
              "legend-item": {
                "background-color": "#007790",
                "borderRadius": "5",
                "font-color": "white"
                },
                "marker": {
                  "background-color": "#da534d",
                  "border-width": 0,
                  "shadow": 0 ,
                  "border-color": "#faa39f"
                },
                "highlight-marker": {
                  "size": 6,
                  "background-color": "#da534d",
                }
              },
              {"values":benchmark_cum,
              "line-color":"#ff0000",
              "line-style":"line",
              "text": "Benchmark Cum %",
              "legend-item": {
                "background-color": "#007790",
                "borderRadius": "5",
                "font-color": "white"
                },
                "marker": {
                  "background-color": "#da534d",
                  "border-width": 0,
                  "shadow": 0 ,
                  "border-color": "#faa39f"
                },
                "highlight-marker": {
                  "size": 6,
                  "background-color": "#da534d",
                }
              },
            ],
          };

          zingchart.render({
            id: 'performance_chart',
            data: myConfig,
          });


			/* ------------------
			---------------------
			Ranking Chart
			---------------------
			--------------------- */

			var rank = [];
			var value_1 = [];
			var value_2 = [];
			var value_3 = [];
			var date_len = date.length-1;


			if(tab_value==tab_value_index[2] + "K")
			{
				var rank_title = "v4micro " + board_value + " Ranking from’ " + date[0] + " to " + date[date_len];

				var day20 = sortObject(result.v4micro_ranking['20Day Lookback']);

                var day5 = result.v4micro_ranking['3Day Lookback'];

                var day1 = result.v4micro_ranking['1Day Lookback'];

                $.each(day20, function(l,m) {

                    var vv=0;
                    $.each(this, function(k, v) {

                        if(k!='-8 Rank RiskOff' && k!='9 Rank RiskOn')
                        {
                            if(vv==0)
                            {
                                
                                rank.push(v);
                                vv=vv+1;

                            }
                            else
                            {
                                
                                value_1.push(v);
                                vv=0;

                            }   
                        }
                    });
                });

                 
                $.each(rank, function(k, v) {
                       
                    $.each(day5, function(o, p) {
                        if(k!='-8 Rank RiskOff' && k!='9 Rank RiskOn')
                        {

                            if(o==v)
                            {

                                  value_2.push(p);

                            }
                        }
                    });

                    $.each(day1, function(o, p) {
                        if(k!='-8 Rank RiskOff' && k!='9 Rank RiskOn')
                        {

                            if(o==v)
                            {

                                  value_3.push(p);

                            }
                        }

                    });
                });
			}
			else if(tab_value==tab_value_index[1] + "K")
			{         
				var rank_title = "v4mini " + board_value + " ‘Ranking from’ " + date[0] + " to " + date[date_len];

				var day20 = sortObject(result.v4mini_ranking['20Day Lookback']);

                var day5 = result.v4mini_ranking['3Day Lookback'];

                var day1 = result.v4mini_ranking['1Day Lookback'];

				$.each(day20, function(l,m) {

                    var vv=0;
                    $.each(this, function(k, v) {

                        if(k!='-4 Rank RiskOff' && k!='4 Rank RiskOn')
                        {
                            if(vv==0)
                            {
                                
                                rank.push(v);
                                vv=vv+1;

                            }
                            else
                            {
                                
                                value_1.push(v);
                                vv=0;

                            }   
                        }
                    });
                });

                 
                $.each(rank, function(k, v) {
                       
                    $.each(day5, function(o, p) {
                    	if(k!='-4 Rank RiskOff' && k!='4 Rank RiskOn')
                        {

                            if(o==v)
                            {

                                
                                  value_2.push(p);

                            }
                        }
                    });

                    $.each(day1, function(o, p) {
                    	if(k!='-4 Rank RiskOff' && k!='4 Rank RiskOn')
                        {

                            if(o==v)
                            {

                                
                                  value_3.push(p);

                            }
                        }

                    });
                });
			}
			else if(tab_value==tab_value_index[0] + "K")
			{      
				var rank_title = "v4futures " + board_value + " ‘Ranking from’ " + date[0] + " to " + date[date_len]; 
                
                var day20 = sortObject(result.v4futures_ranking['20Day Lookback']);

                var day5 = result.v4futures_ranking['3Day Lookback'];

                var day1 = result.v4futures_ranking['1Day Lookback'];
    
				$.each(day20, function(l,m) {

                    var vv=0;
    	            $.each(this, function(k, v) {

    	                if(k!='-20 Rank RiskOff' && k!='20 Rank RiskOn')
    	                {
                            if(vv==0)
                            {
                                
                                rank.push(v);
                                vv=vv+1;

                            }
                            else
                            {
                                
                                value_1.push(v);
                                vv=0;

                            }   
    	                }
    	            });
				});

             
                $.each(rank, function(k, v) {
                       
                    $.each(day5, function(o, p) {
                    	if(k!='-20 Rank RiskOff' && k!='20 Rank RiskOn')
    	                {

	                        if(o==v)
	                        {

	                            
	                              value_2.push(p);

	                        }
	                    }
                    });

                    $.each(day1, function(o, p) {
                    	if(k!='-20 Rank RiskOff' && k!='20 Rank RiskOn')
    	                {

	                        if(o==v)
	                        {

	                            
	                              value_3.push(p);

	                        }
	                    }

                    });
                });
			}

			// console.log("test");
			var data  = [];
            var i = 0;
            count = 0;
            identify_color = "#000000";
            value_1_color = "#0000FF";
            value_2_color = "#F7C143";
            value_3_color = "#0B850C";
            var v1 = '';
            var v2 = '';
            var v3 = '';

            $.each(rank, function(k,v) {

            	i_color = "";
            	identify_color = "#000000";
            	value_1_color = "#0000FF";

            	v = rank_index(v);

            	if(board_value == v   ||   "Anti-" + board_value == v)
            	{
            		i_color = "#FF0000";
            		value_1_color = "#FF0000";
            	}

            	if(v ==  "Previous"  	||  v ==  "Anti-Previous"  	|| v ==  "LowestEquity"	||  v ==  "AntiLowestEquity"||
            	   v ==  "RiskOff"		||  v ==  "RiskOn"  		|| v ==  "Custom"		||  v ==  "Anti-Custom"  	||
            	   v ==  "Seasonality"	||  v ==  "Anti-Seasonality"|| v ==  "HighestEquity"||  v ==  "AntiHighestEquity"||
            	   v ==  "50/50"		||  v ==  "Anti50/50"  		|| v ==  "Off"			||  v ==  "on"  			||
            	   v ==  "benchmark"  )
            	{
            		i_color = "#0000FF";
            	}

            	if(btn_click == 3)
            	{
            		if(value_1[i] == value_1[i-1])
            		{
            			v = v + " (" + count + ")";
            		}
            		else
            		{
            			count = count + 1;
            			v = v + " (" + count + ")";            			
            		}
            	}

        		v1 = Math.abs(value_1[i])
        		v2 = Math.abs(value_2[i])
        		v3 = Math.abs(value_3[i])

            	var arr = [v1,v2,v3];

				var check = indexOf(arr);
			
				if(check[0] == 0 && check[1] == 1 && check[2] == 2 )
				{
					if(i_color !== '')
					{
						identify_color = i_color;
					}

					// console.log("1");

					data.push({
						"category" : v,
						"color" : identify_color,
						"segments" : [{
							"start" : 0,
							"end" : value_1[i],
							"colors" : value_1_color,
							"task": "20Day"
						},{
							"start" : 0,
							"end" : value_2[i],
							"colors" : value_2_color,
							"task": "3Day"
						},{
							"start" : 0,	
							"end" : value_3[i],
							"colors" : value_3_color,
							"task": "1Day"
						}]
					});
					i_color = '';
				}

				else if(check[0] == 0 && check[1] == 2 && check[2] == 1 )
				{
					if(i_color !== '')
					{
						identify_color = i_color;
					}

					// console.log("2");

					data.push({
						"category" : v,
						"color" : identify_color,
						"segments" : [{
							"start" : 0,
							"end" : value_1[i],
							"colors" : value_1_color,
							"task": "20Day"
						},{
							"start" : 0,
							"end" : value_3[i],
							"colors" : value_3_color,
							"task": "1Day"
						},{
							"start" : 0,	
							"end" : value_2[i],
							"colors" : value_2_color,
							"task": "3Day"
						}]
					});
				}

				else if(check[0] == 1 && check[1] == 0 && check[2] == 2 )
				{
					if(i_color !== '')
					{
						identify_color = i_color;
					}

					// console.log("3");

					data.push({
						"category" : v,
						"color" : identify_color,
						"segments" : [{
							"start" : 0,
							"end" : value_2[i],
							"colors" : value_2_color,
							"task": "3Day"
						},{
							"start" : 0,
							"end" : value_1[i],
							"colors" : value_1_color,
							"task": "20Day"
						},{
							"start" : 0,	
							"end" : value_3[i],
							"colors" : value_3_color,
							"task": "1Day"
						}]
					});
				}

				else if(check[0] == 1 && check[1] == 2 && check[2] == 0 )
				{
					if(i_color !== '')
					{
						identify_color = i_color;
					}

					// console.log("4");

					data.push({
						"category" : v,
						"color" : identify_color,
						"segments" : [{
							"start" : 0,
							"end" : value_2[i],
							"colors" : value_2_color,
							"task": "3Day"
						},{
							"start" : 0,
							"end" : value_3[i],
							"colors" : value_3_color,
							"task": "1Day"
						},{
							"start" : 0,	
							"end" : value_1[i],
							"colors" : value_1_color,
							"task": "20Day"
						}]
					});
				}

				else if(check[0] == 2 && check[1] == 0 && check[2] == 1 )
				{
					if(i_color !== '')
					{
						identify_color = i_color;
					}

					// console.log("5");

					data.push({
						"category" : v,
						"color" : identify_color,
						"segments" : [{
							"start" : 0,
							"end" : value_3[i],
							"colors" : value_3_color,
							"task": "1Day"
						},{
							"start" : 0,
							"end" : value_1[i],
							"colors" : value_1_color,
							"task": "20Day"
						},{
							"start" : 0,	
							"end" : value_2[i],
							"colors" : value_2_color,
							"task": "3Day"
						}]
					});
				}

				else if(check[0] == 2 && check[1] == 1 && check[2] == 0 )
				{
					if(i_color !== '')
					{
						identify_color = i_color;
					}

					// console.log("6");

					data.push({
						"category" : v,
						"color" : identify_color,
						"segments" : [{
							"start" : 0,
							"end" : value_3[i],
							"colors" : value_3_color,
							"task": "1Day"
						},{
							"start" : 0,
							"end" : value_2[i],
							"colors" : value_2_color,
							"task": "3Day"
						},{
							"start" : 0,	
							"end" : value_1[i],
							"colors" : value_1_color,
							"task": "20Day"
						}]
					});
				}

                i = i + 1;
            });

			// console.log(data.length);

			var d = new Date();
			rank_startdate = d.getDate()+"-"+(d.getMonth()+1)+"-"+d.getFullYear();
			rank_enddate = d.getDate()+"-"+(d.getMonth()+1)+"-"+d.getFullYear();;


    		AmCharts.addInitHandler( function ( chart ) {

		        // set base values
		        var categoryWidth = 25;
		          
		        // calculate bottom margin based on number of data points
		        var chartHeight = categoryWidth * chart.dataProvider.length;

		        chart.div.style.height = chartHeight + 'px';
        
      		}, ['gantt'] );
			

			

			var chart = AmCharts.makeChart( "ranking_chart", {
				"type": "gantt",
				"theme": "none",
				"titles": [{
					"text": rank_title,
					"size": 16
				}],
				"marginRight": 70,
				"columnWidth": 1,
				"valueAxis": {
					"type": "number"
				},
				"brightnessStep": 10,
				"graph": {
					"fillAlphas": 4,
					"balloonText": "Series '[[task]] Lookback Point [[v]]<br>Value:[[value]]",
          			"pointerWidth": 50,
				},
				"rotate": true,
				"addClassNames": true,
				"categoryField": "category",
				"segmentsField": "segments",
				"colorField": "colors",
				"startDate": "2015-01-01",
				"startField": "start",
				"endField": "end",
				"durationField": "duration",
			    "listeners": [{
					"event": "rendered",
					"method": updateLabels
					},{
				    "event": "resized",
				    "method": updateLabels
			    }],
				"dataProvider": data,
				"chartCursor": {
					"cursorColor":"#55bb76",
					"valueBalloonsEnabled": false,
					"cursorAlpha": 0,
					"valueLineAlpha":0.5,
					"valueLineBalloonEnabled": true,
					"valueLineEnabled": true,
					"zoomable":false,
					"valueZoomable":true
				},  
				"export": {
					"enabled": true
				}
			} );

			function updateLabels(event) { 
	          	var labels = event.chart.chartDiv.getElementsByClassName("amcharts-axis-label");
	          	// console.log(data.length);

	          	for (var i = 0; i < (data.length); i++) {
	            	var color = event.chart.dataProvider[i].color; 
			            if (color !== undefined)  { 
			            	// console.log(color);
			              	labels[i].setAttribute("fill", color);
			            }
	          	}
	        }

			/* ------------------
			---------------------
			Ranking Chart finish
			---------------------
			--------------------- */

        }
    });
});























/* ---------------------------------
------------------------------------
Account_performance_chart Done
------------------------------------
--------------------------------- */

// zingchart.MODULESDIR = "https://cdn.zingchart.com/modules/";
// ZC.LICENSE = ["569d52cefae586f634c54f86dc99e6a9", "ee6b7db5b51705a13dc2339db3edaf6d"];

function getOnlyDate(strr)
{
    var str = new Date(strr);
    var d = str.getDate();
    var m = str.getMonth();
    var y = str.getFullYear();
    
    m = m + 1;

    return y + "-" + m + "-" + d;
}


$(document).on('click', '.chip-button', function(event) {

	$(".chart_loader").show();

    var board_value = $('.chart-title-text').html();
    board_value=board_value.replace(/\s+/g, '');

    var tab_value = $('.chart-icon-text').html();
    tab_value=tab_value.replace(/\s+/g, '');

    var account_value_str = $('.bet-info-table').find('.chip-avatar-text').text();

    var n = account_value_str.match(/K/g);

    var j = 0;

    var account_value = [];

    for(var i = 0; i<n.length; i++)
    {
        var c = account_value_str.search("K");

        var sub = account_value_str.substr(0,c);

        if(account_value[j-1] != sub)
        {
            account_value[j] = sub;
            j = j + 1;
        }
        
        account_value_str = account_value_str.substr(c+1);  
    }

    // console.log(account_value);
    
    var anti_val = '';
    if(board_value == 'AccountPerformanceChart' && tab_value != '')
    {
        $.ajax
        ({
            url: '/getchartdata',
            type: 'get',
            dataType: 'json', // added data type
            success: function(result)
            {   
            	$(".chart_loader").hide();
                /* ------------------
                ---------------------
                Account_performance_chart
                ---------------------
                --------------------- */

                var date = [];

                var benchmark_values = [];
                var benchmark_values_percent = [];
                var benchmark_sym = [];

                var yaxis_values = [];
                var yaxis_values_percent = [];
                var selection = [];
                var slippage = [];
                var commissions = [];

                var simulated_moc_values = [];
                var simulated_moc_values_percent = [];

                if(tab_value == account_value[0] + "K")
                {
                    $.each(result.v4micro_accountvalues, function(l,m) {

                        /*----------------
                        Red Solid Line
                        ------------------*/

                        if(l=='benchmark_values')
                        {
                            $.each(this, function(k, v) {

                                k = getOnlyDate(k);
                                // console.log(k);

                                date.push(k);
                                v = Math.round(v);
                                benchmark_values.push(v);

                            });
                        }

                        if(l=='benchmark_values_percent')
                        {
                            $.each(this, function(k, v) {

                                v = v * 100;
                                v = parseFloat(v).toFixed(2);
                                v = v + " %";
                                benchmark_values_percent.push(v);

                            });
                        }

                        if(l=='benchmark_sym')
                        {
                            $.each(this, function(k, v) {

                                benchmark_sym.push(v);

                            });
                        }

                        /*----------------
                        Blue Solid Line
                        ------------------*/

                        if(l=='yaxis_values')
                        {
                            $.each(this, function(k, v) {

                                v = Math.round(v);
                                yaxis_values.push(v);

                            });
                        }

                        if(l=='yaxis_values_percent')
                        {
                            $.each(this, function(k, v) {

                                v = v * 100;
                                v = parseFloat(v).toFixed(2);
                                v = v + " %"
                                yaxis_values_percent.push(v);

                            });
                        }

                        if(l=='selection')
                        {
                            $.each(this, function(k, v) {

                                selection.push(v);

                            });
                        }

                        if(l=='slippage')
                        {
                            $.each(this, function(k, v) {

                                v = Math.round(v);
                                slippage.push(v);

                            });
                        }

                        if(l=='commissions')
                        {
                            $.each(this, function(k, v) {

                                v = Math.round(v);
                                commissions.push(v);

                            });
                        }

                        /*----------------
                        Green Solid Line
                        ------------------*/

                        if(l=='simulated_moc_values')
                        {
                            $.each(this, function(k, v) {

                                v = Math.round(v);
                                simulated_moc_values.push(v);

                            });
                        }

                        if(l=='simulated_moc_values_percent')
                        {
                            $.each(this, function(k, v) {

                                v = v * 100;
                                v = parseFloat(v).toFixed(2);
                                v = v + " %"
                                simulated_moc_values_percent.push(v);

                            });
                        }
                        
                    });
                }

                else if(tab_value == account_value[1] + "K")
                {
                    $.each(result.v4mini_accountvalues, function(l,m) {

                        /*----------------
                        Red Solid Line
                        ------------------*/

                        if(l=='benchmark_values')
                        {
                            $.each(this, function(k, v) {

                                k = getOnlyDate(k);
                                // console.log(k);

                                date.push(k);
                                v = Math.round(v);
                                benchmark_values.push(v);

                            });
                        }

                        if(l=='benchmark_values_percent')
                        {
                            $.each(this, function(k, v) {

                                v = v * 100;
                                v = parseFloat(v).toFixed(2);
                                v = v + " %";
                                benchmark_values_percent.push(v);

                            });
                        }

                        if(l=='benchmark_sym')
                        {
                            $.each(this, function(k, v) {

                                benchmark_sym.push(v);

                            });
                        }

                        /*----------------
                        Blue Solid Line
                        ------------------*/

                        if(l=='yaxis_values')
                        {
                            $.each(this, function(k, v) {

                                v = Math.round(v);
                                yaxis_values.push(v);

                            });
                        }

                        if(l=='yaxis_values_percent')
                        {
                            $.each(this, function(k, v) {

                                v = v * 100;
                                v = parseFloat(v).toFixed(2);
                                v = v + " %"
                                yaxis_values_percent.push(v);

                            });
                        }

                        if(l=='selection')
                        {
                            $.each(this, function(k, v) {

                                selection.push(v);

                            });
                        }

                        if(l=='slippage')
                        {
                            $.each(this, function(k, v) {

                                v = Math.round(v);
                                slippage.push(v);

                            });
                        }

                        if(l=='commissions')
                        {
                            $.each(this, function(k, v) {

                                v = Math.round(v);
                                commissions.push(v);

                            });
                        }

                        /*----------------
                        Green Solid Line
                        ------------------*/

                        if(l=='simulated_moc_values')
                        {
                            $.each(this, function(k, v) {

                                v = Math.round(v);
                                simulated_moc_values.push(v);

                            });
                        }

                        if(l=='simulated_moc_values_percent')
                        {
                            $.each(this, function(k, v) {

                                v = v * 100;
                                v = parseFloat(v).toFixed(2);
                                v = v + " %"
                                simulated_moc_values_percent.push(v);

                            });
                        }
                    });
                }

                else if(tab_value == account_value[2] + "K")
                {
                    $.each(result.v4futures_accountvalues, function(l,m) {

                        /*----------------
                        Red Solid Line
                        ------------------*/

                        if(l=='benchmark_values')
                        {
                            $.each(this, function(k, v) {

                                k = getOnlyDate(k);
                                // console.log(k);

                                date.push(k);
                                v = Math.round(v);
                                benchmark_values.push(v);

                            });
                        }

                        if(l=='benchmark_values_percent')
                        {
                            $.each(this, function(k, v) {

                                v = v * 100;
                                v = parseFloat(v).toFixed(2);
                                v = v + " %";
                                benchmark_values_percent.push(v);

                            });
                        }

                        if(l=='benchmark_sym')
                        {
                            $.each(this, function(k, v) {

                                benchmark_sym.push(v);

                            });
                        }

                        /*----------------
                        Blue Solid Line
                        ------------------*/

                        if(l=='yaxis_values')
                        {
                            $.each(this, function(k, v) {

                                v = Math.round(v);
                                yaxis_values.push(v);

                            });
                        }

                        if(l=='yaxis_values_percent')
                        {
                            $.each(this, function(k, v) {

                                v = v * 100;
                                v = parseFloat(v).toFixed(2);
                                v = v + " %"
                                yaxis_values_percent.push(v);

                            });
                        }

                        if(l=='selection')
                        {
                            $.each(this, function(k, v) {

                                selection.push(v);

                            });
                        }

                        if(l=='slippage')
                        {
                            $.each(this, function(k, v) {

                                v = Math.round(v);
                                slippage.push(v);

                            });
                        }

                        if(l=='commissions')
                        {
                            $.each(this, function(k, v) {

                                v = Math.round(v);
                                commissions.push(v);

                            });
                        }

                        /*----------------
                        Green Solid Line
                        ------------------*/

                        if(l=='simulated_moc_values')
                        {
                            $.each(this, function(k, v) {

                                v = Math.round(v);
                                simulated_moc_values.push(v);

                            });
                        }

                        if(l=='simulated_moc_values_percent')
                        {
                            $.each(this, function(k, v) {

                                v = v * 100;
                                v = parseFloat(v).toFixed(2);
                                v = v + " %"
                                simulated_moc_values_percent.push(v);

                            });
                        }
                    });
                }
                else
                {
                    console.log("NO BOARD");
                }

                /*----------------
                Chart Title
                ------------------*/

                if(tab_value == account_value[0] + "K")
                {
                    chart_title = "v4 micro Equity Chart " + date.length + " Day Lookback";
                }
                else if(tab_value == account_value[1] + "K")
                {
                    chart_title = "v4mini Equity Chart " + date.length + " Day Lookback";
                }
                else if(tab_value == account_value[2] + "K")
                {
                    chart_title = "v4future Equity Chart " + date.length + " Day Lookback";
                }
                else
                {
                    console.log("No Title");
                }

                var max_benchmark_values = Math.max.apply(Math,benchmark_values);
                var min_benchmark_values = Math.min.apply(Math,benchmark_values);

                var max_yaxis_values = Math.max.apply(Math,yaxis_values);
                var min_yaxis_values = Math.min.apply(Math,yaxis_values);

                var max_simulated_moc_values = Math.max.apply(Math,simulated_moc_values);
                var min_simulated_moc_values = Math.min.apply(Math,simulated_moc_values);

                var max = Math.max(max_benchmark_values, max_yaxis_values, max_simulated_moc_values);
                var min = Math.min(min_benchmark_values, min_yaxis_values, min_simulated_moc_values);

                max = Math.round(max) + 5000;
                min = Math.round(min) - 5000;

                var tot='';
                tot+=min.toString();
                tot+=":";
                tot+=max.toString();
                tot+=":1000";

                var account_performance =  {
                  "type":"line",
                  "utc": true,
                  "title": {
                    "text": chart_title,
                    "font-size": "14px",
                    "adjust-layout": true
                  },
                  "plotarea": {
                    "margin": "dynamic 45 60 dynamic",
                  },
                  "scale-x":{
                    "values":date,
                    "transform": {
                      "type": "date",
                      "all": "%d %M %Y",
                      "guide": {
                        "visible": false
                      },
                    },
                    "item":{  
                      "font-angle":315,  
                    } 
                  },
                  "scale-y":{
                    "values":tot,
                    "guide": {
                      "line-style": "dashed"
                    },
                    "thousands-separator":",",
                  },
                  "plot": {
                    "highlight": true,
                    "tooltip-text": "%t: %v<br>Date:%k",
                    "shadow": 0,
                    "line-width": "2px",
                    "marker": {
                      "type": "circle",
                      "size": 1
                    },
                    "highlight-state": {
                      "line-width": 3
                    },
                    "animation": {
                      "effect": 1,
                      "sequence": 2,
                      "speed": 100,
                    },
                  },
                  "crosshair-x": {
                    "line-color": "#efefef",
                    "plot-label": {
                      "border-radius": "5px",
                      "border-width": "1px",
                      "border-color": "#f6f7f8",
                      "padding": "10px",
                      "font-weight": "bold",
                      "thousands-separator":",",
                      "font-size": "15px",
                    },
                    "scale-label": {
                      "font-color": "#000",
                      "background-color": "#f6f7f8",
                      "border-radius": "5px"
                    },
                  },
                  "tooltip": {
                    "visible": false
                  },
                  "series":[
                    {"values":benchmark_values,
                    "line-color":"#FF00FF",
                    "line-style":"line",
                    "text": "Benchmark Value ($)",
                    "legend-item": {
                      "background-color": "#007790",
                      "borderRadius": "5",
                      "font-color": "white"
                      },
                      "marker": {
                        "background-color": "#da534d",
                        "border-width": 0,
                        "shadow": 0,
                        "border-color": "#faa39f"
                      },
                      "highlight-marker": {
                        "size": 6,
                        "background-color": "#da534d",
                      }
                    },
                    {"values":benchmark_values_percent,
                    "line-color":"#FF00FF",
                    "line-style":"line",
                    "text": "Benchmark % Chg",
                    "legend-item": {
                      "background-color": "#007790",
                      "borderRadius": "5",
                      "font-color": "white"
                      },
                      "marker": {
                        "background-color": "#da534d",
                        "border-width": 0,
                        "shadow": 0 ,
                        "border-color": "#faa39f"
                      },
                      "highlight-marker": {
                        "size": 6,
                        "background-color": "#da534d",
                      }
                    },
                    {"values":benchmark_sym,
                    "line-color":"#FF00FF",
                    "line-style":"line",
                    "text": "Symbol",
                    "legend-item": {
                      "background-color": "#007790",
                      "borderRadius": "5",
                      "font-color": "white"
                      },
                      "marker": {
                        "background-color": "#da534d",
                        "border-width": 0,
                        "shadow": 0 ,
                        "border-color": "#faa39f"
                      },
                      "highlight-marker": {
                        "size": 6,
                        "background-color": "#da534d",
                      }
                    },
                    {"values":yaxis_values,
                    "line-color":"#0000FF",
                    "line-style":"line",
                    "text": "Account Value ($)",
                    "legend-item": {
                      "background-color": "#007790",
                      "borderRadius": "5",
                      "font-color": "white"
                      },
                      "marker": {
                        "background-color": "#da534d",
                        "border-width": 0,
                        "shadow": 0 ,
                        "border-color": "#faa39f"
                      },
                      "highlight-marker": {
                        "size": 6,
                        "background-color": "#da534d",
                      }
                    },
                    {"values":yaxis_values_percent,
                    "line-color":"#0000FF",
                    "line-style":"line",
                    "text": "Account % Chg",
                    "legend-item": {
                      "background-color": "#007790",
                      "borderRadius": "5",
                      "font-color": "white"
                      },
                      "marker": {
                        "background-color": "#da534d",
                        "border-width": 0,
                        "shadow": 0 ,
                        "border-color": "#faa39f"
                      },
                      "highlight-marker": {
                        "size": 6,
                        "background-color": "#da534d",
                      }
                    },
                    {"values":selection,
                    "line-color":"#0000FF",
                    "line-style":"line",
                    "text": "Bet",
                    "legend-item": {
                      "background-color": "#007790",
                      "borderRadius": "5",
                      "font-color": "white"
                      },
                      "marker": {
                        "background-color": "#da534d",
                        "border-width": 0,
                        "shadow": 0 ,
                        "border-color": "#faa39f"
                      },
                      "highlight-marker": {
                        "size": 6,
                        "background-color": "#da534d",
                      }
                    },
                    {"values":slippage,
                    "line-color":"#0000FF",
                    "line-style":"line",
                    "text": "Slippage ($)",
                    "legend-item": {
                      "background-color": "#007790",
                      "borderRadius": "5",
                      "font-color": "white"
                      },
                      "marker": {
                        "background-color": "#da534d",
                        "border-width": 0,
                        "shadow": 0 ,
                        "border-color": "#faa39f"
                      },
                      "highlight-marker": {
                        "size": 6,
                        "background-color": "#da534d",
                      }
                    },
                    {"values":commissions,
                    "line-color":"#0000FF",
                    "line-style":"line",
                    "text": "Commission ($)",
                    "legend-item": {
                      "background-color": "#007790",
                      "borderRadius": "5",
                      "font-color": "white"
                      },
                      "marker": {
                        "background-color": "#da534d",
                        "border-width": 0,
                        "shadow": 0 ,
                        "border-color": "#faa39f"
                      },
                      "highlight-marker": {
                        "size": 6,
                        "background-color": "#da534d",
                      }
                    },
                    {"values":simulated_moc_values,
                    "line-color":"#0B850C",
                    "line-style":"line",
                    "text": "Simulated MOC Value ($)",
                    "legend-item": {
                      "background-color": "#007790",
                      "borderRadius": "5",
                      "font-color": "white"
                      },
                      "marker": {
                        "background-color": "#da534d",
                        "border-width": 0,
                        "shadow": 0 ,
                        "border-color": "#faa39f"
                      },
                      "highlight-marker": {
                        "size": 6,
                        "background-color": "#da534d",
                      }
                    },
                    {"values":simulated_moc_values_percent,
                    "line-color":"#0B850C",
                    "line-style":"line",
                    "text": "MOC % Chg",
                    "legend-item": {
                      "background-color": "#007790",
                      "borderRadius": "5",
                      "font-color": "white"
                      },
                      "marker": {
                        "background-color": "#da534d",
                        "border-width": 0,
                        "shadow": 0 ,
                        "border-color": "#faa39f"
                      },
                      "highlight-marker": {
                        "size": 6,
                        "background-color": "#da534d",
                      }
                    },
                  ],
                };

                zingchart.render({
                  id: 'account_performance_chart',
                  data: account_performance,
                });

                /* ---------------------------------
                ------------------------------------
                Account_performance_chart Done
                ------------------------------------
                --------------------------------- */
            }
        });
    }
});