"""Fig. 4: keeps panels (a) and (b) of the published drawing (../data/fig04_panels_ab.*) and redraws panel (c), the table of selected configurations.
Run from this folder: python fig4c.py  -> fig04.svg, fig04.png"""
import os
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt, re
from PIL import Image
plt.rcParams.update({'font.family':'serif','font.serif':['Liberation Serif'],'svg.fonttype':'path'})
W=518.4; H=128.0
fig=plt.figure(figsize=(W/72,H/72)); ax=fig.add_axes([0,0,1,1]); ax.set_xlim(0,W); ax.set_ylim(H,0); ax.axis('off')
fs=6.6
ax.plot([4,W-4],[2,2],color='#9E9E9E',lw=0.6)
ax.text(4,13,'(c) Selected configurations (Tables II and XIII)',fontsize=7.8,fontweight='bold',va='center')
cols=[4,74,128,252,316,382,430,470]
hdr=['Model','Input','CNN branch','Readout','LSTM','Branch dense','Head','Parameters']
for x,h in zip(cols,hdr): ax.text(x,27,h,fontsize=fs,fontweight='bold',va='center')
rows=[['DS05, parallel','W = 30, C = 28','2 blocks: 192, 96; k = 5; p = 0','global avg pool','1 × 88','none','8','162,001'],
      ['DS05, sequential','W = 30, C = 28','4 blocks: 120, 63, 34, 18; k = 2; p = 2','—','1 × 128','—','201, 69','142,784'],
      ['FD003, parallel','W = 50, C = 25','1 block: 128; k = 3; p = 1','flatten','2 × 128','128 | 128','256','712,449'],
      ['FD003, sequential','W = 50, C = 25','1 block: 128; k = 3; p = 1','—','2 × 128','—','128, 256','—'],
      ['B0005, parallel','W = 10, C = 1','1 block: 3 × 32 (k = 3, 5, 7); p = 1','global avg pool','1 × 32, bidir.','none','64','19,649'],
      ['B0005, sequential','W = 10, C = 1','1 block: 3 × 32 (k = 3, 5, 7); p = 1','—','1 × 32, bidir.','—','64','37,825']]
for i,r in enumerate(rows):
    y=40+i*11.2
    for j,(x,c) in enumerate(zip(cols,r)): ax.text(x,y,c,fontsize=fs,va='center',fontweight='bold' if j==0 else None)
ax.text(4,40+6*11.2+3,'B0005: the network models the residual about a linear trend fitted to the 20 cycles before the forecast origin (Eq. 19). '
        'LSTM layers are unidirectional unless marked bidir.; p = pooled blocks.',fontsize=6.0,va='center',style='italic')
fig.savefig('panel_c.svg'); fig.savefig('panel_c.png',dpi=1382/(W/72)); plt.close(fig)
# ---- compose SVG
old=open(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'data', 'fig04_panels_ab.svg')).read()
m=re.search(r'viewBox="0 0 ([\d.]+) ([\d.]+)"',old); w0,h0=float(m.group(1)),float(m.group(2))
cut=293.0; newH=cut+H
pc=open('panel_c.svg').read()
pc=pc[pc.index('<svg'):]
pc=re.sub(r'id="([^"]+)"',r'id="pc_\1"',pc); pc=re.sub(r'href="#([^"]+)"',r'href="#pc_\1"',pc); pc=re.sub(r'url\(#([^)]+)\)',r'url(#pc_\1)',pc)
pc=re.sub(r'<svg ',' <svg x="0" y="%.2f" '%cut,pc,count=1)
pc=re.sub(r'(<svg[^>]*?) width="[^"]*pt" height="[^"]*pt"',r'\1 width="%.2f" height="%.2f"'%(W,H),pc,count=1)
new=old.replace('viewBox="0 0 %s %s"'%(m.group(1),m.group(2)),'viewBox="0 0 %s %.4f"'%(m.group(1),newH),1)
new=re.sub(r'height="[\d.]+pt"','height="%.4fpt"'%newH,new,count=1)
k=new.rindex('</svg>')
new=new[:k]+'<rect x="0" y="%.2f" width="%.2f" height="%.2f" fill="#ffffff"/>'%(cut-1,w0,h0-cut+2)+pc+'\n'+new[k:]
open('fig04.svg','w').write(new)
# ---- compose PNG
im=Image.open(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'data', 'fig04_panels_ab.png')).convert('RGB'); sc=im.size[0]/w0
top=im.crop((0,0,im.size[0],int(cut*sc)))
pcp=Image.open('panel_c.png').convert('RGB').resize((im.size[0],int(H*sc)))
out=Image.new('RGB',(im.size[0],top.size[1]+pcp.size[1]),'white'); out.paste(top,(0,0)); out.paste(pcp,(0,top.size[1])); out.save('fig04.png')
print(w0,h0,newH,out.size)
