"""Assemble current vector panels with unchanged archived schematics."""
from pathlib import Path
import base64
import copy
import re
import xml.etree.ElementTree as ET

ROOT=Path(__file__).resolve().parents[1]
NS='http://www.w3.org/2000/svg';XL='http://www.w3.org/1999/xlink'
ET.register_namespace('',NS);ET.register_namespace('xlink',XL)
def tag(x):return '{'+NS+'}'+x

def svgpath(n,stem):
    p=ROOT/f'manuscript/figure_{n}/figure_files/panels'
    candidates=list(p.rglob(stem+'.svg'));assert len(candidates)==1,(n,stem,candidates);return candidates[0]

def add_svg(parent,path,box,prefix):
    x,y,w,h=box;node=ET.parse(path).getroot();ids={e.attrib['id']:prefix+'_'+e.attrib['id'] for e in node.iter() if 'id' in e.attrib}
    for e in node.iter():
        for k,v in list(e.attrib.items()):
            if k=='id':e.set(k,ids[v]);continue
            for old,new in ids.items():
                v=v.replace('url(#'+old+')','url(#'+new+')')
                if k.endswith('href') and v=='#'+old:v='#'+new
            e.set(k,v)
    node.set('x',str(x));node.set('y',str(y));node.set('width',str(w));node.set('height',str(h));node.set('preserveAspectRatio','xMidYMid meet')
    parent.append(node)

def add_image(parent,path,box):
    x,y,w,h=box;ET.SubElement(parent,tag('image'),{'x':str(x),'y':str(y),'width':str(w),'height':str(h),'preserveAspectRatio':'xMidYMid meet','{'+XL+'}href':'data:image/png;base64,'+base64.b64encode(path.read_bytes()).decode()})

def letter(parent,s,x,y):ET.SubElement(parent,tag('text'),{'x':str(x),'y':str(y),'font-family':'Arial','font-size':'38','font-weight':'bold','fill':'#111111'}).text=s

def extract_schematics():
    existing={(n,key):ROOT/f'manuscript/figure_{n}/figure_files/schematics/{key}.png'
              for n,keys in {1:['a','b','d','e','f'],2:['a','c'],3:['a','b'],4:['b','e']}.items() for key in keys}
    return existing

def main():
    schemes=extract_schematics()
    sizes={1:(2400,2820),2:(3000,2400),3:(3000,2800),4:(2800,2800),5:(2800,3160)}
    layouts={
      1:[('a','scheme',(0,0,1000,510)),('b','scheme',(1020,0,1380,510)),('c','Figure1C.aida',(0,530,800,480)),('','Figure1C.covid_normal',(800,530,800,480)),('','Figure1C.covid_patient',(1600,530,800,480)),('d','scheme',(0,1030,2400,790)),('e','scheme',(0,1840,1290,980)),('f','scheme',(1310,1840,1090,980))],
      2:[('a','scheme',(0,0,750,650)),('b','Figure2B.mean_expression_correlation',(770,0,1460,650)),('c','scheme',(2250,0,750,650)),('d','Figure2D.global_de_pearson_r',(0,680,1480,750)),('e','Figure2E.global_top_de_overlap',(1520,680,1480,750)),('f','Figure2F.global_sign_agreement',(0,1460,1000,850)),('g','Figure2G.cell_type_de_pearson_r',(1010,1460,990,850)),('h','Figure2H.model_category_rankings',(2010,1460,990,850))],
      3:[('a','scheme',(0,0,1000,650)),('b','scheme',(1020,0,1980,650)),('c','Figure3C.age_real_vs_predicted',(0,680,1780,1230)),('d','Figure3D.localization_mae_r',(1800,680,1200,550)),('e','Figure3E.cell_type_heatmap',(1800,1250,1200,690)),('f','Figure3F.cellular_senescence',(0,1970,1200,800)),('g','Figure3G.tnf_nfkb',(1210,1970,1200,800)),('h','Figure3H.bcr_tcr_signaling',(2420,1970,580,800))],
      4:[('a','Figure4A.age_acceleration_5y_forest',(0,0,1400,760)),('b','scheme',(1420,0,1380,760)),('c','Figure4C.gap_slope_dotplot',(0,790,1350,1120)),('d','Figure4D.aging_alignment_scatter',(1380,790,1420,1120)),('e','scheme',(0,1940,1350,860)),('f','Figure4F.orthogonal_displacement_decomposition',(1380,1940,1420,860))],
      5:[('a','Figure5A.AD_EN_program_heatmap',(0,0,1210,1100)),('b','Figure5B.AD_EN_subclass_shared_contribution',(1420,0,1380,1000)),('c','Figure5C.COVID_program_heatmap',(0,1130,1600,850)),('d','Figure5D.COVID_program_dispersion',(1770,1130,900,850)),('e','Figure5E.COVID_IFN_symptom_onset',(0,2010,850,1080)),('f','Figure5F.COVID_monocyte_subtype_remodeling',(880,2010,1920,900))]
    }
    legend_slots=[('Figure5A.AD_EN_program_heatmap__legend',(1210,140,180,660)),('Figure5B.AD_EN_subclass_shared_contribution__legend',(1500,1000,1200,100)),('Figure5C.COVID_program_heatmap__legend',(1600,1300,160,550)),('Figure5D.COVID_program_dispersion__legend',(2670,1300,130,550)),('Figure5E.COVID_IFN_symptom_onset__legend',(0,3050,850,100)),('Figure5F.COVID_monocyte_subtype_remodeling__legend',(1000,2890,1700,150))]
    for n,(width,height) in sizes.items():
        root=ET.Element(tag('svg'),{'width':str(width),'height':str(height),'viewBox':f'0 0 {width} {height}'})
        ET.SubElement(root,tag('rect'),{'width':'100%','height':'100%','fill':'white'})
        for index,(key,stem,box) in enumerate(layouts[n]):
            x,y,w,h=box;content=(x+10,y+45,w-20,h-50)
            if stem=='scheme':add_image(root,schemes[n,key],content)
            else:add_svg(root,svgpath(n,stem),content,f'Figure{n}_{index}')
            if key:letter(root,key,x+8,y+36)
        if n==5:
            for index,(stem,box) in enumerate(legend_slots):add_svg(root,svgpath(5,stem),box,f'Figure5_legend{index}')
        dest=ROOT/f'manuscript/figure_{n}/figure_files/figure{n}_sccody.svg';ET.ElementTree(root).write(dest,encoding='utf-8',xml_declaration=True)
        print(dest,flush=True)

if __name__=='__main__':main()
