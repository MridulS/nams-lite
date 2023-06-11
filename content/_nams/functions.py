import numpy as np
from markdown import Markdown
from IPython.display import HTML




def render_html(ans):
    md = Markdown()
    return HTML(md.convert(ans))
