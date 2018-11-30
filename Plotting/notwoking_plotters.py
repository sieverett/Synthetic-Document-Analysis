import mpld3
import pandas as pd

#define custom toolbar location
class TopToolbar(mpld3.plugins.PluginBase):
    """Plugin for moving toolbar to top of figure"""

    JAVASCRIPT = """
    mpld3.register_plugin("toptoolbar", TopToolbar);
    TopToolbar.prototype = Object.create(mpld3.Plugin.prototype);
    TopToolbar.prototype.constructor = TopToolbar;
    function TopToolbar(fig, props){
        mpld3.Plugin.call(this, fig, props);
    };

    TopToolbar.prototype.draw = function(){
      // the toolbar svg doesn't exist
      // yet, so first draw it
      this.fig.toolbar.draw();

      // then change the y position to be
      // at the top of the figure
      this.fig.toolbar.toolbar.attr("x", 150);
      this.fig.toolbar.toolbar.attr("y", 400);

      // then remove the draw function,
      // so that it is not called again
      this.fig.toolbar.draw = function() {}
    }
    """
    def __init__(self):
        self.dict_ = {"type": "toptoolbar"}

def d3_plot_clusters(x,y,clusters,titles):

    """
    Plots a d3 scatter plot with tooltip.
    
    
    args:
    x - an np.array of x values of data points
    y - an np.array of y values of data points
    clusters - an np.array of cluster centers
    titles - an array of strings for annotating each data point
    
    cluster_colors = {0: '#1b9e77', 1: '#d95f02', 2: '#7570b3', 3: '#e7298a', 4: '#66a61e',
                  5: '#d85f02', 6: '#6570b3', 7: '#e6298a', 8: '#56a61e', 9: '#e9a61e'}
    """
    
    
    
    df = pd.DataFrame(dict(x=x, y=y, label=clusters, title=titles)) 

    #group by cluster, with options to sample 
    # n = 300  
    #groups = df.sample(n).groupby('label')
    groups = df.groupby('label')

    #define custom css to format the font and to remove the axis labeling
    css = """
    text.mpld3-text, div.mpld3-tooltip {
      font-family:Arial, Helvetica, sans-serif;
    }
    
    g.mpld3-xaxis, g.mpld3-yaxis {
    display: none; }
    """

# Plot 
    fig, ax = plt.subplots(figsize=(14,10)) #set plot size
    ax.margins(0.03) # Optional, just adds 5% padding to the autoscaling

#iterate through groups to layer the plot
    # add labels for the legend with label=cluster_names[name]

    for name, group in groups:
        points = ax.plot(group.x, group.y, marker='o', linestyle='', ms=18, mec='none', color=cluster_colors[name])
        ax.set_aspect('auto')
        labels = [i for i in group.title]
    
        #set tooltip using points, labels and the already defined 'css'
        tooltip = mpld3.plugins.PointHTMLTooltip(points[0], labels,
                                       voffset=10, hoffset=10, css=css)
        #connect tooltip to fig
        mpld3.plugins.connect(fig, tooltip, TopToolbar())    
    
        #set tick marks as blank
        ax.axes.get_xaxis().set_ticks([])
        ax.axes.get_yaxis().set_ticks([])
    
        #set axis as blank
        ax.axes.get_xaxis().set_visible(False)
        ax.axes.get_yaxis().set_visible(False)

    
    ax.legend(numpoints=1,loc='lower left') #show legend with only one dot

    return mpld3.display() #show the plot

    #uncomment the below to export to html
    # html = mpld3.fig_to_html(fig)
    #print(html)