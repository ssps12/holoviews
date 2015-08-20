from ...core import Store, Overlay, NdOverlay, Layout, AdjointLayout, GridSpace
from ...element import Curve, Points, Scatter, Image, Raster, Path, RGB, Histogram, Spread, HeatMap
from ...element import Contours, Path, Box, Bounds, Ellipse, Polygons, ErrorBars, Text, HLine, VLine
from ...interface.seaborn import Bivariate, TimeSeries
from ...core.options import Options, Cycle, OptionTree
from ..plot import PlotSelector
from ..mpl.seaborn import TimeSeriesPlot, BivariatePlot

from .annotation import TextPlot, LineAnnotationPlot
from .element import OverlayPlot, BokehMPLWrapper
from .chart import PointPlot, CurvePlot, SpreadPlot, ErrorPlot, LinkedScatter, LinkedScatterPlot, HistogramPlot
from .path import PathPlot, PolygonPlot
from .plot import GridPlot, LayoutPlot, AdjointLayoutPlot
from .raster import RasterPlot, RGBPlot
from .renderer import BokehRenderer

Store.renderers['bokeh'] = BokehRenderer

def wrapper(obj):
    return 'bokeh'

Store.register({Overlay: OverlayPlot,
                NdOverlay: OverlayPlot,
                Curve: CurvePlot,
                Points: PointPlot,
                Scatter: PointPlot,
                Spread: SpreadPlot,
                HLine: LineAnnotationPlot,
                VLine: LineAnnotationPlot,
                GridSpace: GridPlot,
                LinkedScatter: LinkedScatterPlot,
                Image: RasterPlot,
                RGB: RGBPlot,
                Raster: RasterPlot,
                HeatMap: RasterPlot,
                Histogram: HistogramPlot,
                AdjointLayout: AdjointLayoutPlot,
                Layout: LayoutPlot,
                Path: PathPlot,
                TimeSeries: PlotSelector(wrapper, [('mpl', TimeSeriesPlot), ('bokeh', BokehMPLWrapper)], True),
                Bivariate: PlotSelector(wrapper, [('mpl', BivariatePlot), ('bokeh', BokehMPLWrapper)], True),
                Contours: PathPlot,
                Path:     PathPlot,
                Box:      PathPlot,
                Bounds:   PathPlot,
                Ellipse:  PathPlot,
                Polygons: PolygonPlot,
                ErrorBars: ErrorPlot,
                Text: TextPlot}, 'bokeh')

Cycle.default_cycles['default_colors'] =  ['#30a2da', '#fc4f30', '#e5ae38',
                                           '#6d904f', '#8b8b8b']

options = Store.options(backend='bokeh')

# Charts
options.Curve = Options('style', color=Cycle(), line_width=2)
options.Scatter = Options('style', color=Cycle())
options.ErrorBars = Options('style', color='k')
options.Spread = Options('style', fill_color=Cycle(), fill_alpha=0.6, line_color='black')
options.LinkedScatter = Options('style', size=12, color=Cycle(),
                                marker=Cycle(values=['circle', 'square', 'triangle',
                                                     'diamond', 'inverted_triangle']))
options.Histogram = Options('style', fill_color="#036564", line_color="#033649")
options.Points = Options('style', color=Cycle())

# Paths
options.Contours = Options('style', color=Cycle())
options.Path = Options('style', color=Cycle())
options.Box = Options('style', color='black')
options.Bounds = Options('style', color='black')
options.Ellipse = Options('style', color='black')
options.Polygons = Options('style', color=Cycle())

# Rasters
options.Image = Options('style', cmap='hot')
options.Raster = Options('style', cmap='hot')
options.QuadMesh = Options('style', cmap='hot')
options.HeatMap = Options('style', cmap='RdYlBu_r')

