"""Conceptual publication diagram: no empirical measurements or numeric labels."""
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch


def main():
    fig, ax = plt.subplots(figsize=(11, 5))
    ax.set(xlim=(0, 1), ylim=(0, 1))
    ax.axis('off')
    def box(x, y, text, color, width=.28, height=.16):
        ax.add_patch(FancyBboxPatch((x-width/2,y-height/2),width,height,
            boxstyle='round,pad=0.005,rounding_size=0.018',facecolor=color,
            edgecolor='#4b5966',linewidth=1.1))
        ax.text(x,y,text,ha='center',va='center',fontsize=11,color='#17212b')
    def arrow(a,b):
        ax.add_patch(FancyArrowPatch(a,b,arrowstyle='->',mutation_scale=14,
                                     linewidth=1.4,color='#66717e'))
    box(.18,.84,'Planned\nconfigurations','#dcecf8')
    box(.50,.84,'Measure\naccuracy + time','#e8f1e9')
    box(.82,.84,'Fit response\nmodels','#e8f1e9')
    arrow((.33,.84),(.35,.84));arrow((.65,.84),(.67,.84))
    box(.35,.53,'Quadratic surface\nANOVA effects','#e8f1e9',.32)
    box(.77,.53,'Flexible surface\nSHAP explanations','#f0eafa',.32)
    arrow((.77,.75),(.35,.62));arrow((.82,.75),(.77,.62))
    box(.35,.23,'Compare + select\nnew configurations','#fdece8',.32)
    box(.77,.23,'Reserved\nconfirmation','#fff2df',.32)
    arrow((.35,.44),(.35,.32));arrow((.71,.44),(.45,.32))
    arrow((.52,.23),(.60,.23))
    ax.plot([.18,.01,.01,.02],[.23,.23,.84,.84],color='#66717e',lw=1.4)
    arrow((.02,.84),(.03,.84))
    ax.text(.02,.02,'Repeat measurements and refit; confirmation observations remain outside the fit.',
            fontsize=10,color='#4b5966',transform=ax.transAxes)
    path=Path(__file__).resolve().parents[1]/'figures/explainer_sequential_extension.pdf'
    fig.savefig(path,bbox_inches='tight',metadata={'CreationDate':None,'ModDate':None})
    plt.close(fig)


if __name__ == '__main__':
    main()
