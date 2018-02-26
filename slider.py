''' Present an interactive function explorer with slider widgets.
Scrub the sliders to change the properties of the ``sin`` curve, or
type into the title text box to update the title of the plot.
Use the ``bokeh serve`` command to run the example by executing:
    bokeh serve sliders.py
at your command prompt. Then navigate to the URL
    http://localhost:5006/sliders
in your browser.
'''
import numpy as np

from bokeh.io import curdoc
from bokeh.layouts import row, widgetbox
from bokeh.models import ColumnDataSource
from bokeh.models.widgets import Slider, TextInput
from bokeh.plotting import figure
from bokeh.models import HoverTool

from scipy import stats
import csv



#Read in fit parameters
with open("sv_params.csv",'r') as csvfile:
    reader = csv.reader(csvfile, delimiter=' ',
                            quotechar='|', quoting=csv.QUOTE_MINIMAL)
    for readrow in reader:
        fit_params = np.array(readrow).astype(float)
print("Got Here")
# Set up data
N = 100
darkrate = np.linspace(-2.5, 1.0, N)
year = 2001.5
#model
loc_slope1,loc_int1,loc_slope2,loc_int2,scale_slope1,scale_int1,scale_slope2,scale_int2,shape_slope1,shape_int1,shape_slope2,shape_int2,c,d=fit_params #darkrate dependence and time dep parameters
dt_arr = year
#Time dependent parameters
loc = np.piecewise(dt_arr,
                    [(dt_arr < 2010.3),(dt_arr>2010.3)],
                    [lambda dt_arr:loc_slope1*dt_arr+loc_int1,
                    lambda dt_arr: loc_slope2*dt_arr+loc_int2])
scale = np.piecewise(dt_arr,
                    [(dt_arr <= 2010.3),(dt_arr>2010.3)],
                    [lambda dt_arr:scale_slope1*dt_arr+scale_int1,
                    lambda dt_arr:scale_slope2*dt_arr+scale_int2])
shape = np.piecewise(dt_arr,
                    [(dt_arr <= 2010.3),(dt_arr>2010.3)],
                    [lambda dt_arr:shape_slope1*dt_arr+shape_int1,
                    lambda dt_arr:shape_slope2*dt_arr+shape_int2])
#Rate dependent factor
t = (darkrate-loc) / scale
sv_curve = 2 / scale * stats.norm.pdf(t) * stats.norm.cdf(shape*t) + c*darkrate + d

source = ColumnDataSource(data=dict(x=darkrate, y=sv_curve))

hover = HoverTool()

# Set up plot
plot = figure(plot_height=600, plot_width=800, title="Scale Value Curve",
              tools="crosshair,pan,reset,save,wheel_zoom,hover",
              x_range=[-2.65, 1.15], y_range=[0.04, 0.11])

plot.line('x', 'y', source=source, line_width=3, line_alpha=0.6)


# Set up widgets

text = TextInput(title="title", value='Scale Value Curve')
#offset = Slider(title="offset", value=0.0, start=-5.0, end=5.0, step=0.1)
#amplitude = Slider(title="amplitude", value=1.0, start=-5.0, end=5.0, step=0.1)
#phase = Slider(title="phase", value=0.0, start=0.0, end=2*np.pi)
#freq = Slider(title="frequency", value=1.0, start=0.1, end=5.1, step=0.1)
year_slider = Slider(title="Time",value=2001.5, start = 2001.5, end=2017.5, step=0.117)


# Set up callbacks
def update_title(attrname, old, new):
    plot.title.text = text.value

text.on_change('value', update_title)

def update_data(attrname, old, new):

    # Get the current slider values
    #a = amplitude.value
    #b = offset.value
    #w = phase.value
    #k = freq.value
    year = year_slider.value

    # Generate the new curve
    N = 100
    darkrate = np.linspace(-2.5, 1.0, N)

    loc_slope1,loc_int1,loc_slope2,loc_int2,scale_slope1,scale_int1,scale_slope2,scale_int2,shape_slope1,shape_int1,shape_slope2,shape_int2,c,d=fit_params #darkrate dependence and time dep parameters
    dt_arr = year
    #Time dependent parameters
    loc = np.piecewise(dt_arr,
                        [(dt_arr < 2010.3),(dt_arr>2010.3)],
                        [lambda dt_arr:loc_slope1*dt_arr+loc_int1,
                        lambda dt_arr: loc_slope2*dt_arr+loc_int2])
    scale = np.piecewise(dt_arr,
                        [(dt_arr <= 2010.3),(dt_arr>2010.3)],
                        [lambda dt_arr:scale_slope1*dt_arr+scale_int1,
                        lambda dt_arr:scale_slope2*dt_arr+scale_int2])
    shape = np.piecewise(dt_arr,
                        [(dt_arr <= 2010.3),(dt_arr>2010.3)],
                        [lambda dt_arr:shape_slope1*dt_arr+shape_int1,
                        lambda dt_arr:shape_slope2*dt_arr+shape_int2])
    #Rate dependent factor
    t = (darkrate-loc) / scale
    sv_curve = 2 / scale * stats.norm.pdf(t) * stats.norm.cdf(shape*t) + c*darkrate + d

    source.data = dict(x=darkrate, y=sv_curve)

for w in [year_slider]:
    w.on_change('value', update_data)


# Set up layouts and add to document
inputs = widgetbox(text, year_slider)

row(inputs, plot, width=800)
curdoc().add_root(row(inputs, plot, width=800))
curdoc().title = "Sliders"
