"""Report figures from saved fields and time histories; PNG only."""
from pathlib import Path
import json
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from model import ROOT,parameters,make_grid,pump_profile,spin_stationary
from numerics import load_0d,observables_0d

WIDTH=160/25.4
STYLE={'figure.dpi':120,'font.family':'DejaVu Sans','font.size':9,
       'axes.titlesize':10,'axes.labelsize':9,'legend.fontsize':8,
       'xtick.labelsize':8,'ytick.labelsize':8,'axes.grid':True,
       'grid.alpha':.25,'grid.linewidth':.6,'axes.linewidth':.8,'lines.linewidth':1.5,
       'figure.constrained_layout.w_pad':.025,'figure.constrained_layout.h_pad':.045}
COLORS={'M0':'0.18','R1':'tab:blue','R2':'tab:orange'}
SCALAR_LABELS={'M0':'One reservoir (M0)','R1':'Two reservoirs (R1)','R2':'Two reservoirs (R2)'}
SCALAR_STYLES={'M0':'-','R1':(0,(5,2)),'R2':(0,(3,1.5,1,1.5))}
FIGURE_HEIGHTS_MM={'F1_reservoir_response':100,'F2_ring_reservoir_comparison':105,
                   'F3_spin_pump_response':78,'F4_internal_josephson_dynamics':92,
                   'F5_josephson_regime_scan':120}
DATA=ROOT/'results/data'


def _layout_check(fig,name):
    """Check the physical canvas; do not crop fonts or resize after export."""
    from matplotlib.text import Text
    fig.canvas.draw();fig.canvas.draw();renderer=fig.canvas.get_renderer();bounds=fig.bbox
    axes=list(fig.axes)
    for ax in list(axes):axes.extend(ax.child_axes)
    texts=list(fig.texts)
    for ax in axes:
        texts.extend([ax.title,ax._left_title,*ax.texts])
        if not ax.axison:continue
        texts.extend([ax.xaxis.label,ax.yaxis.label])
        for axis,limits in ((ax.xaxis,ax.get_xlim()),(ax.yaxis,ax.get_ylim())):
            for tick in axis.get_major_ticks():
                if min(limits)<=tick.get_loc()<=max(limits):texts.extend([tick.label1,tick.label2])
            texts.append(axis.get_offset_text())
    for legend in [*fig.legends,*(ax.get_legend() for ax in axes)]:
        if legend is not None:texts.extend(legend.findobj(Text))
    boxes=[];seen=set()
    for artist in texts:
        if id(artist) in seen or not artist.get_visible() or not artist.get_text():continue
        seen.add(id(artist));box=artist.get_window_extent(renderer)
        if not box.width or not box.height:continue
        if box.x0<-1 or box.y0<-1 or box.x1>bounds.x1+1 or box.y1>bounds.y1+1:
            raise ValueError(f'Clipped text in {name}: {artist.get_text()}')
        boxes.append((artist,box))
    for i,(first,a) in enumerate(boxes):
        for second,b in boxes[i+1:]:
            if min(a.x1,b.x1)-max(a.x0,b.x0)>2 and min(a.y1,b.y1)-max(a.y0,b.y0)>2:
                raise ValueError(f'Overlapping text in {name}: {first.get_text()} / {second.get_text()}')


def save(fig,out,name,*,width_mm=160):
    out=Path(out);out.mkdir(parents=True,exist_ok=True)
    height=FIGURE_HEIGHTS_MM.get(name,fig.get_size_inches()[1]*25.4)
    fig.set_size_inches(width_mm/25.4,height/25.4);fig.set_dpi(300)
    _layout_check(fig,name)
    path=out/(name+'.png');fig.savefig(path,dpi=300,facecolor='white',pil_kwargs={'optimize':True})
    plt.close(fig)
    return path


def _scalar_legend(fig,title):
    handles=[Line2D([],[],color=COLORS[k],ls=SCALAR_STYLES[k],label=SCALAR_LABELS[k]) for k in ('M0','R1','R2')]
    fig.legend(handles=handles,loc='outside upper center',ncol=3,frameon=False,
               title=title,title_fontsize=8.5,handlelength=2.7,columnspacing=1.2)


def figure1(out):
    fig,axes=plt.subplots(2,2,figsize=(WIDTH,100/25.4),layout='constrained')
    for name in ('M0','R1','R2'):
        data=load_0d(DATA/f'F1_{name}.npz');obs=observables_0d(data);t=data['time']
        style={'color':COLORS[name],'ls':SCALAR_STYLES[name]}
        axes[0,0].plot(t,obs['nc'][:,0],**style)
        axes[0,1].plot(t,obs['active'][:,0],**style)
        if name=='M0':axes[1,0].step(t-1000,obs['feed'][:,0],where='post',**style)
        else:axes[1,0].plot(t-1000,obs['feed'][:,0],**style)
        axes[1,1].plot(t-1000,obs['nc'][:,0],**style)
        p=parameters(data['config'])
    for ax,value in ((axes[0,0],.5*p['Pth']/p['gammaC']),(axes[0,1],p['nAth'])):
        ax.axhline(value,color='.6',ls=':',lw=.8)
    for ax in axes[0]:ax.set_xlim(0,400);ax.set_xticks([0,100,200,300,400]);ax.set_xlabel('Time after pump on (ps)')
    for ax in axes[1]:
        ax.set_xlim(-10,100);ax.set_xticks([0,50,100]);ax.axvline(0,color='.5',ls=':',lw=.9)
        ax.set_xlabel('Time relative to pump off (ps)')
    for ax,title,label in zip(axes.flat,
            ('(a) Condensate formation','(b) Active reservoir','(c) Residual reservoir feeding','(d) Condensate decay'),
            (r'$n_C$ ($\mu$m$^{-2}$)',r'$n_A$ or $n_R$ ($\mu$m$^{-2}$)',
             r'$Q_A$ ($\mu$m$^{-2}$ ps$^{-1}$)',r'$n_C$ ($\mu$m$^{-2}$)')):
        ax.set_title(title,loc='left');ax.set_ylabel(label);ax.set_ylim(bottom=-.02)
    axes[1,1].set_ylim(-.02,1.27)
    _scalar_legend(fig,r'Pump $P_A=1.5P_{\rm th}$; pump off at $t=1000$ ps')
    return save(fig,out,'F1_reservoir_response')


def figure3(out):
    records=json.loads((DATA/'F3_scan.json').read_text());eps=np.array([r['epsilon'] for r in records])
    n=np.array([r['density'] for r in records]);config=load_0d(DATA/'F3_eps_+0.0.npz')['config'];p=parameters(config)
    fine=np.linspace(-1,1,201);analytic=np.array([spin_stationary(2,e,p)[0] for e in fine])
    fig,axes=plt.subplots(1,3,figsize=(WIDTH,78/25.4),layout='constrained')
    for j,label in enumerate((r'$+$ circular',r'$-$ circular')):
        axes[0].plot(fine,analytic[:,j],color=f'C{j}',lw=1.3)
        axes[0].plot(eps,n[:,j],'o',ms=3.1,mfc='white',color=f'C{j}',label=label)
    for key,color in [('z','C0'),('SA','C1'),('SI','C2')]:
        axes[1].plot(eps,[r[key] for r in records],'o',ms=3.1,mfc='white',color=color)
    di=(p['gammaI']+p['W'])/(p['gammaI']+p['W']+2*p['GammaI'])
    axes[1].plot(fine,2*di*fine,color='C0',lw=1.3);axes[1].plot(fine,di*fine,color='C2',lw=1.3)
    axes[1].axhline(0,color='C1',lw=1.3)
    ideal=np.array([spin_stationary(2,e,p)[3] for e in fine])
    axes[2].plot(fine,1000*(ideal[:,0]-ideal[:,1]),color='C0',lw=1.3)
    axes[2].plot(eps,1000*np.array([r['energy_difference'] for r in records]),'o',ms=3.1,mfc='white',color='C0')
    for ax,title in zip(axes,('(a) Condensate densities','(b) Spin polarizations','(c) Energy difference')):
        ax.set_title(title,loc='left');ax.set_xlabel(r'Pump polarization, $\epsilon$');ax.set_xticks([-1,0,1])
    axes[0].set_ylabel(r'$n_\pm$ ($\mu$m$^{-2}$)');axes[1].set_ylabel('Circular polarization')
    axes[2].set_ylabel(r'$\hbar(\omega_+-\omega_-)$ ($\mu$eV)');axes[2].set_yticks([-40,-20,0,20,40])
    axes[0].legend(frameon=False,loc='upper center',handlelength=.6)
    polarization_handles=[Line2D([],[],color='C0',label=r'Condensate $z$'),
                          Line2D([],[],color='C1',label=r'Active $S_A$'),
                          Line2D([],[],color='C2',label=r'Inactive $S_I$')]
    axes[1].legend(handles=polarization_handles,frameon=False,loc='lower right',handlelength=1.2,
                   borderpad=.2,labelspacing=.25)
    handles=[Line2D([],[],color='.3',label='Steady-state formula'),
             Line2D([],[],ls='',marker='o',mfc='white',color='.3',label='Time integration',ms=3.1)]
    fig.legend(handles=handles,loc='outside upper center',ncol=2,frameon=False,
               title=r'No coherent coupling ($J=0$); total pump $P_{A,\mathrm{tot}}=4P_{\rm th}$ (R1)',title_fontsize=8.5,
               handlelength=1.6,columnspacing=1.1)
    return save(fig,out,'F3_spin_pump_response')


def figure4(out):
    """One regime per column, one observable per row; no overlapping cases."""
    fig,axes=plt.subplots(2,3,figsize=(WIDTH,92/25.4),layout='constrained',sharex=True)
    cases=((.005,'C1','Running phase'),(.0065,'C2','Oscillating phase'),(.010,'C0','Fixed phase'))
    for col,(J,color,regime) in enumerate(cases):
        data=load_0d(DATA/f'F5_J_{J:.5f}.npz');obs=observables_0d(data)
        mask=(data['time']>=4000)&(data['time']<=6000);t=data['time'][mask]
        theta=obs['theta'][mask]-obs['theta'][mask][0]
        axes[0,col].plot(t,obs['z'][mask],color=color,lw=1.4)
        axes[1,col].plot(t,theta,color=color,lw=1.4)
        axes[0,col].set_title(f'({chr(97+col)}) {regime}\n'+rf'$J={1000*J:g}$ $\mu$eV',loc='left',fontsize=9)
        axes[1,col].set_title(f'({chr(100+col)}) Phase change',loc='left',fontsize=9)
        axes[0,col].set_ylim(-.3,1.08);axes[0,col].set_yticks([0,.5,1.])
        if col==0:
            axes[0,col].set_ylabel('Spin imbalance, $z$')
        else:axes[0,col].tick_params(labelleft=False)
        axes[1,col].set_ylabel(r'$\Delta\theta$ (rad)')
        axes[1,col].set_ylim((-39,2) if col==0 else (-.3,2.5))
        axes[1,col].set_yticks([-30,-15,0] if col==0 else [0,1,2])
        axes[1,col].set_xlabel('Time (ps)')
        for ax in axes[:,col]:
            ax.set_xlim(4000,6000);ax.set_xticks([4000,5000,6000]);ax.axhline(0,color='.65',lw=.65,zorder=0)
    fig.suptitle(r'Spinor R1: $P_{A,\mathrm{tot}}=4P_{\rm th}$, $\epsilon=0.3$',fontsize=9)
    return save(fig,out,'F4_internal_josephson_dynamics')


def figure5(out):
    rows=sorted(json.loads((DATA/'F5_scan.json').read_text()),key=lambda r:r['J'])
    styles={'UNCOUPLED':('o','.25','No coupling'),
            'RUNNING_PHASE_PERIODIC':('^','C1','Running phase'),
            'BOUNDED_PHASE_PERIODIC':('v','C2','Oscillating phase'),
            'LOCKED_STEADY':('s','C0','Fixed phase'),
            'PERIODIC_PHASE_MASKED':('D','.5','Phase masked'),
            'UNRESOLVED':('D','.5','Unresolved')}
    fig,axes=plt.subplots(3,2,figsize=(WIDTH,120/25.4),layout='constrained')
    handles=[]
    for status,(marker,color,label) in styles.items():
        group=[r for r in rows if r['status']==status]
        if not group:continue
        handles.append(Line2D([],[],ls='',marker=marker,color=color,label=label,ms=4))
        for record in group:
            x=1000*record['J']
            for row,key in enumerate(('plot_mean_z','A_z','mean_phase_drift')):
                fallback={'plot_mean_z':'mean_z','mean_phase_drift':'phase_slope'}.get(key,key)
                value=record.get(key,record.get(fallback))
                if value is None or not np.isfinite(value):continue
                for col in range(2):
                    axes[row,col].plot(x,value,ls='',marker=marker,ms=3.1,color=color,mew=.6)
    for row in range(3):
        ax=axes[row,0];ax.set_xlim(-.5,20.5);ax.set_xticks([0,5,10,15,20])
        if row<2:ax.tick_params(labelbottom=False)
    axes[0,1].set_xlim(11.2,12.2);axes[0,1].set_xticks([11.2,11.7,12.2])
    axes[0,1].set_xlabel(r'$J$ ($\mu$eV)',labelpad=1)
    for ax in axes[1:,1]:
        ax.set_xlim(6.1,7.1);ax.set_xticks([6.1,6.6,7.1])
    axes[1,1].tick_params(labelbottom=False)
    for ax in axes[0]:ax.set_ylim(-.65,.85);ax.axhline(0,color='.6',lw=.7)
    for ax in axes[1]:ax.set_ylim(-.025,.68)
    axes[2,0].axhline(0,color='.6',ls=':',lw=.8)
    axes[2,1].clear()
    for record in rows:
        marker,color,_=styles[record['status']]
        axes[2,1].plot(1000*record['J'],record['A_z'],ls='',marker=marker,ms=3.1,color=color,mew=.6)
    axes[2,1].set_xlim(11.45,11.7);axes[2,1].set_xticks([11.5,11.6,11.7]);axes[2,1].set_ylim(-.025,.52)
    axes[2,1].set_ylabel(r'$A_z$')
    for ax,title in zip(axes.flat,('(a) Mean spin imbalance','(b) Polarization reversal',
                                 '(c) Oscillation amplitude','(d) Oscillations near locking',
                                 '(e) Mean phase drift','(f) Cycles near reversal')):
        ax.set_title(title,loc='left')
    axes[0,0].set_ylabel(r'Mean $z$')
    axes[1,0].set_ylabel(r'Amplitude, $A_z$')
    axes[2,0].set_ylabel('Phase drift\n(rad ps$^{-1}$)')
    for ax in axes[2]:ax.set_xlabel(r'Coherent coupling, $J$ ($\mu$eV)')
    check_file=DATA/'reference_checks.json'
    checks=json.loads(check_file.read_text()) if check_file.exists() else {}
    controls=checks.get('F5_polarization_controls',{}).get('continuations',[])
    if controls:
        for record in controls:
            axes[0,1].plot(1000*record['J'],record['plot_mean_z'],'o',ms=4.5,mfc='none',mec='.15',mew=.8)
        axes[0,1].text(.5,.52,'Open circles:\ncontinued states',transform=axes[0,1].transAxes,
                       ha='center',va='center',fontsize=8)
    fig.legend(handles=handles,loc='outside upper center',ncol=3 if len(handles)>4 else 4,
               frameon=False,title=rf'Spinor R1: $P_{{A,\mathrm{{tot}}}}=4P_{{\rm th}}$, $\epsilon=0.3$; {len(rows)} independent starts',
               title_fontsize=8.5,handlelength=1,columnspacing=1.1,labelspacing=.25)
    return save(fig,out,'F5_josephson_regime_scan')

def load_ring_case(name):
    path=ROOT/'results/runs'/name
    metadata=json.loads((DATA/'F2_summary.json').read_text())['cases'][name]['summary']
    if (path/'checkpoint.npz').exists():
        import yaml
        config=yaml.safe_load((path/'resolved_config.yaml').read_text())
        with np.load(path/'checkpoint.npz',allow_pickle=False) as z:
            return config,tuple(z[k].copy() for k in ('psi','active','inactive')),z['history'].copy(),metadata
    with np.load(DATA/'report_data.npz',allow_pickle=False) as z:
        return (json.loads(str(z[name+'_config'])),tuple(z[name+'_'+k].copy() for k in ('psi','active','inactive')),
                z[name+'_history'].copy(),metadata)


def figure2(out):
    """Formation and final density: the same observable for all three models."""
    import matplotlib.patheffects as effects
    cases=[load_ring_case(name) for name in ('F2_M0','F2_R1','F2_R2')]
    times=[c[3]['time_ps'] for c in cases]
    if len(set(times))!=1:raise ValueError('F2 needs a common comparison time')
    fig=plt.figure(figsize=(WIDTH,105/25.4),layout='constrained')
    gs=fig.add_gridspec(2,4,width_ratios=(1,1,1,.055),height_ratios=(.75,1.2))
    population=fig.add_subplot(gs[0,:3])
    for name,(_,_,history,_) in zip(('M0','R1','R2'),cases):
        population.plot(history[:,0],history[:,1],color=COLORS[name],ls=SCALAR_STYLES[name])
    population.set_title('(a) Condensate population',loc='left')
    population.set(xlabel='Time (ps)',ylabel=r'$N_C$',xlim=(0,times[0]),ylim=(0,335))
    population.set_xticks([0,500,1000,1500,2000]);population.set_yticks([0,150,300])
    peak=max(float(np.max(abs(state[0])**2)) for _,state,_,_ in cases)
    for col,((config,state,_,summary),title) in enumerate(zip(cases,('(b) M0','(c) M1–R1','(d) M1–R2'))):
        ax=fig.add_subplot(gs[1,col]);grid=make_grid(config);density=abs(state[0][0])**2
        extent=(grid.x[0]-grid.dx/2,grid.x[-1]+grid.dx/2,grid.y[0]-grid.dx/2,grid.y[-1]+grid.dx/2)
        im=ax.imshow(density,origin='lower',extent=extent,cmap='magma',vmin=0,vmax=peak,interpolation='nearest')
        contour=ax.contour(grid.x,grid.y,pump_profile(config,grid),levels=[.5],colors='white',linewidths=.8,linestyles='--')
        contour.set_path_effects([effects.Stroke(linewidth=1.35,foreground='.1'),effects.Normal()])
        ax.set(xlim=(-20,20),ylim=(-20,20),aspect='equal',xlabel=r'$x$ ($\mu$m)')
        ax.set_xticks([-20,0,20]);ax.set_yticks([-20,0,20]);ax.grid(False);ax.set_title(title,loc='left')
        if col==0:ax.set_ylabel(r'$y$ ($\mu$m)')
        else:ax.tick_params(labelleft=False)
        ax.text(.95,.95,f"$q={summary['vortex']['q']:+d}$",ha='right',va='top',fontsize=9,color='white',transform=ax.transAxes)
    cax=fig.add_subplot(gs[1,3]);cax.grid(False)
    bar=fig.colorbar(im,cax=cax,ticks=[0,.2,.4,.6]);bar.set_label(r'$n_C$ ($\mu$m$^{-2}$)')
    fig.supxlabel('Dashed circles: half-maximum pump contour',fontsize=8)
    _scalar_legend(fig,rf'Ring pumping; final density at $t={times[0]:g}$ ps')
    return save(fig,out,'F2_ring_reservoir_comparison')


def diagnostic2(out):
    fig,axes=plt.subplots(5,3,figsize=(WIDTH,8.5),layout='constrained')
    records=json.loads((DATA/'F5_scan.json').read_text())
    for col,J in enumerate((.005,.0065,.010)):
        data=load_0d(DATA/f'F5_J_{J:.5f}.npz');obs=observables_0d(data);t=data['time'];final=t[-1]
        record=min(records,key=lambda r:abs(r['J']-J))
        for row,(values,label) in enumerate(((obs['z'],'$z$'),(obs['theta'],r'$\theta$ (rad)'),(obs['IJ'],r'$I_J$ ($\mu$m$^{-2}$ ps$^{-1}$)'))):
            axes[row,col].plot(t,values,color=f'C{row}');axes[row,col].set_ylabel(label)
            axes[row,col].set_xlim(0,final)
            if row<2:axes[row,col].tick_params(labelbottom=False)
        ends=np.arange(1000,final+1,500);amplitudes=[]
        for end in ends:
            mask=(t>=end-1000)&(t<=end);amplitudes.append(np.ptp(obs['z'][mask])/2)
        axes[3,col].plot(ends,amplitudes,'o-',ms=3);axes[3,col].set_ylabel('$A_z$ in 1000 ps')
        axes[3,col].set_xlabel('Time (ps)');axes[0,col].set_title(f'$J={1000*J:g}$ '+r'$\mu$eV',loc='left')
        ax=axes[4,col]
        if 'period_ps' in record:
            period=record['period_ps'];left,right=record['window_ps'];mask=(t>=left)&(t+period<=right);base=t[mask]
            n=np.column_stack((obs['nc'],obs['active'],obs['inactive']))
            returned=np.column_stack([np.interp(base+period,t,c) for c in n.T])
            scale=np.sqrt(np.mean(n[(t>=left)&(t<=right)]**2,axis=0))
            error=np.max(abs(returned-n[mask])/np.maximum(scale,1e-12),axis=1)
            phase=np.exp(1j*obs['theta'])
            shifted=np.interp(base+period,t,phase.real)+1j*np.interp(base+period,t,phase.imag)
            ax.semilogy(base,np.maximum(error,1e-16),label='Densities')
            ax.semilogy(base,np.maximum(abs(shifted-phase[mask]),1e-16),label='Relative phase')
            ax.axhline(.01,color='.4',ls=':');ax.legend(frameon=False,fontsize=8)
            ax.set(xlabel='Time (ps)',ylabel='Period-return error')
        else:
            ax.set_axis_off()
            ax.text(.5,.5,'Fixed-phase state\nNo cycle test needed',ha='center',va='center',fontsize=8,transform=ax.transAxes)
    return save(fig,out,'D2_settling_and_phase')


def diagnostic3(out):
    file=DATA/'reference_checks.npz'
    with np.load(file,allow_pickle=False) as z:a={k:z[k].copy() for k in z.files}
    fig,axes=plt.subplots(2,2,figsize=(WIDTH,4.2),layout='constrained')
    axes[0,0].semilogy(a['F3_epsilon'],np.maximum(a['F3_error'],1e-16),'o-',ms=3)
    axes[0,0].set(xlabel=r'$\epsilon$',ylabel='Relative density error');axes[0,0].set_title('(a) F3 vs analytical branch',loc='left')
    for variant in ('R1','R2'):
        axes[0,1].plot(a['Q_time'],a['Q_error_'+variant],label=variant,color=COLORS[variant])
    axes[0,1].set(xlabel='Time (ps)',ylabel=r'$\Delta n_A$ ($\mu$m$^{-2}$)');axes[0,1].legend(frameon=False)
    axes[0,1].set_title('(b) Reservoir-source check',loc='left')
    if 'rabi_time' in a:
        axes[1,0].plot(a['rabi_time'],a['rabi_expected'],color='.3',label='Analytical')
        axes[1,0].plot(a['rabi_time'][::30],a['rabi_density'][::30],'o',mfc='white',ms=3,label='RK4')
        axes[1,0].set(xlabel='Time (ps)',ylabel='$n_+$');axes[1,0].legend(frameon=False)
        axes[1,1].semilogy(a['rabi_time'],np.maximum(a['rabi_error'],1e-16),color='C0')
        axes[1,1].set(xlabel='Time (ps)',ylabel='Absolute field error')
    axes[1,0].set_title('(c) Conservative Rabi test',loc='left');axes[1,1].set_title('(d) Rabi residual',loc='left')
    return save(fig,out,'D3_reference_checks')


def diagnostic1(out):
    report=json.loads((DATA/'F2_summary.json').read_text())
    fig,axes=plt.subplots(3,2,figsize=(WIDTH,5.8),layout='constrained')
    for label,name in zip(('M0','R1','R2'),('F2_M0','F2_R1','F2_R2')):
        c,_,h,s=load_ring_case(name);p=parameters(c)
        total=h[:,1:4].sum(axis=1);residual=abs(total-total[0]-h[:,4])/np.maximum(h[:,5],1e-30)
        axes[0,0].semilogy(h[1:,0],np.maximum(residual[1:],1e-16),label=label,color=COLORS[label])
        component=report.get('component_balance',{}).get(name)
        if component:
            for j,ls in enumerate(('--',':')):
                axes[0,0].semilogy(component['time_ps'][1:],np.maximum(np.array(component['relative'])[1:,j],1e-16),color=COLORS[label],ls=ls,lw=1)
        path=ROOT/'results/runs'/name/'diagnostics.json'
        if path.exists():slow=json.loads(path.read_text())
        else:slow=report['diagnostic_series'][name]
        tt=np.array([r['time'] for r in slow])
        axes[0,1].plot(tt,[r['minimum_inactive'] for r in slow],color=COLORS[label],ls='--')
        for ax,key in ((axes[0,1],'minimum_active'),(axes[1,0],'edge_mass'),(axes[1,1],'high_k')):
            values=[r[key] for r in slow]
            if key.startswith('minimum'):ax.plot(tt,values,color=COLORS[label],label=label)
            else:ax.semilogy(tt,np.maximum(values,1e-16),color=COLORS[label],label=label)
    axes[0,0].axhline(1e-3,color='.4',ls=':');axes[1,0].axhline(1e-4,color='.4',ls=':')
    axes[1,1].axhline(1e-6,color='.4',ls=':')
    titles=('(a) Balance: total / C / A','(b) Minimum reservoir density','(c) Edge population fraction','(d) High-k fraction')
    for ax,title in zip(axes[:2].flat,titles):ax.set_title(title,loc='left');ax.set_xlabel('Time (ps)')
    axes[0,0].set_ylabel('Relative balance residual')
    axes[0,1].set_ylabel(r'Density ($\mu$m$^{-2}$)')
    axes[1,0].set_ylabel('Edge fraction')
    axes[1,1].set_ylabel('High-$k$ fraction')
    axes[0,0].legend(frameon=False)
    comparisons=report['comparisons'];keys=list(comparisons);x=np.arange(len(keys))
    for j,(metric,label) in enumerate((('NC_relative','$N_C$'),('NA_relative','$N_A$'),('density_relative','Density map'))):
        axes[2,0].bar(x+(j-1)*.23,[100*comparisons[k][metric] for k in keys],width=.23,label=label)
    comparison_labels=[{'grid256':r'$256^2$ grid','dt_half':'Half time step'}.get(k,k) for k in keys]
    axes[2,0].set_xticks(x,comparison_labels);axes[2,0].set_ylabel('Relative difference (%)')
    axes[2,0].set_title('(e) Final-state convergence',loc='left');axes[2,0].legend(frameon=False,fontsize=7)
    axes[2,1].bar(x,[100*comparisons[k]['transient_NC_RMSE_relative'] for k in keys],color='C0')
    axes[2,1].set_xticks(x,comparison_labels);axes[2,1].set_ylabel('Relative RMSE (%)')
    axes[2,1].set_title('(f) Population transient',loc='left')
    return save(fig,out,'D1_balance_and_resolution')


def make_figures(*,output_dir=None,diagnostics=True):
    out=ROOT/'results/figures' if output_dir is None else Path(output_dir)
    paths=[]
    with plt.rc_context(STYLE):
        for function in (figure1,figure3,figure4,figure5):paths.append(function(out))
        report_file=DATA/'F2_summary.json'
        if report_file.exists() and json.loads(report_file.read_text()).get('status')=='COMPLETED':
            paths.append(figure2(out))
        if diagnostics:
            diagnostic_out=ROOT/'results/diagnostics' if output_dir is None else out/'diagnostics'
            paths.append(diagnostic2(diagnostic_out))
            if (DATA/'reference_checks.npz').exists():paths.append(diagnostic3(diagnostic_out))
            if report_file.exists() and json.loads(report_file.read_text()).get('status')=='COMPLETED':
                paths.append(diagnostic1(diagnostic_out))
    return paths


# Simplified figures for the next report. Original F1–F5/D1–D3 remain unchanged.
FIGURE_HEIGHTS_MM.update({'N1_reservoir_switching':74,'N2_circular_populations':74,
                         'N3_spatial_components':140,'N4_component_dynamics':82,
                         'N5_relative_phase':82,'N6_coupling_summary':78})
SPIN_COLORS=('C0','C1')


def make_spatial_spin_gif(output_dir=None,*,fps=5,refresh=False):
    """Animate saved N3 component densities without interpolation or integration."""
    from PIL import Image
    import hashlib
    import yaml
    from numerics import write_json

    if not np.isfinite(fps) or fps<=0:raise ValueError('fps must be positive and finite')
    case='N3_eps_03';archive=DATA/(case+'_animation.npz')
    if refresh or not archive.exists():
        run=ROOT/'results/runs'/case
        config=yaml.safe_load((run/'resolved_config.yaml').read_text())
        if config['pump']['epsilon']!=.3 or config['model']!='M2':
            raise ValueError('Animation requires the uncoupled epsilon=0.3 N3 run')
        frames=[];sources=[];x=None;y=None
        for path in sorted((run/'snapshots').glob('*.npz')):
            with np.load(path,allow_pickle=False) as z:
                density=abs(z['psi'])**2
                if density.shape[0]!=2 or not np.isfinite(density).all():
                    raise ValueError('Invalid component densities in '+str(path))
                ix=np.flatnonzero(abs(z['x'])<=20);iy=np.flatnonzero(abs(z['y'])<=20)
                xx=z['x'][ix];yy=z['y'][iy]
                if x is not None and (not np.array_equal(x,xx) or not np.array_equal(y,yy)):
                    raise ValueError('Animation snapshots must use the same grid')
                x,y=xx.copy(),yy.copy()
                da=float((z['x'][1]-z['x'][0])*(z['y'][1]-z['y'][0]))
                frames.append((float(z['time']),density[:,iy[:,None],ix].copy(),density.sum(axis=(-2,-1))*da))
            sources.append({'path':str(path.relative_to(ROOT)),'sha256':hashlib.sha256(path.read_bytes()).hexdigest()})
        if len(frames)<2:raise ValueError('At least two saved spatial snapshots are required')
        order=np.argsort([f[0] for f in frames]);frames=[frames[k] for k in order];sources=[sources[k] for k in order]
        times=np.array([f[0] for f in frames])
        if np.any(np.diff(times)<=0):raise ValueError('Snapshot times must increase strictly')
        DATA.mkdir(parents=True,exist_ok=True);temporary=archive.with_suffix('.tmp.npz')
        np.savez_compressed(temporary,time=times,density=np.stack([f[1] for f in frames]),
                            populations=np.stack([f[2] for f in frames]),x=x,y=y,
                            config_json=json.dumps(config),sources_json=json.dumps(sources))
        temporary.replace(archive)
    with np.load(archive,allow_pickle=False) as z:
        times=z['time'];density=z['density'];populations=z['populations'];x=z['x'];y=z['y']
        sources=json.loads(str(z['sources_json']))
    peak=float(density.max())
    if not np.isfinite(density).all() or peak<=0:raise ValueError('Invalid animation density scale')
    out=ROOT/'results/figures/report' if output_dir is None else Path(output_dir)
    out.mkdir(parents=True,exist_ok=True)
    palette=Image.new('P',(1,1))
    colors=np.vstack((plt.get_cmap('viridis')(np.linspace(0,1,224))[:,:3],
                      np.repeat(np.linspace(0,1,32)[:,None],3,axis=1)))
    palette.putpalette(np.round(255*colors).astype('uint8').ravel().tolist())
    dx=x[1]-x[0];dy=y[1]-y[0]
    extent=(x[0]-dx/2,x[-1]+dx/2,y[0]-dy/2,y[-1]+dy/2)
    rendered=[]
    with plt.rc_context(STYLE):
        fig,axes=plt.subplots(1,2,figsize=(8.5,4.1),dpi=110,layout='constrained')
        images=[]
        for k,ax in enumerate(axes):
            label='+' if k==0 else '-'
            im=ax.imshow(density[0,k],origin='lower',extent=extent,cmap='viridis',
                         vmin=0,vmax=peak,interpolation='nearest')
            images.append(im);ax.grid(False)
            ax.set(xlim=(-20,20),ylim=(-20,20),aspect='equal',
                   xlabel=r'$x$ ($\mu$m)',ylabel=r'$y$ ($\mu$m)')
            ax.set_title(rf'$n_{label}=|\psi_{label}|^2$',fontsize=11)
            ax.set_xticks([-20,0,20]);ax.set_yticks([-20,0,20])
            cb=fig.colorbar(im,ax=ax,shrink=.78,pad=.03)
            cb.set_label(rf'$n_{label}$ ($\mu$m$^{{-2}}$)')
        try:
            for j,t in enumerate(times):
                for k,im in enumerate(images):im.set_data(density[j,k])
                fig.suptitle(rf'Ring pump $\cdot$ $\epsilon=0.3$ $\cdot$ $t={t:g}$ ps',fontsize=13)
                _layout_check(fig,'N3 animation')
                frame=Image.fromarray(np.asarray(fig.canvas.buffer_rgba())[:,:,:3])
                rendered.append(frame.quantize(palette=palette,dither=Image.Dither.NONE))
                frame.close()
            durations=[max(10,round(1000/fps/10)*10)]*len(rendered)
            durations[0]=max(10,round(4*1000/fps/10)*10)
            durations[-1]=max(10,round(18*1000/fps/10)*10)
            path=out/(case+'_evolution.gif')
            temporary=path.with_suffix('.tmp.gif')
            rendered[0].save(temporary,save_all=True,append_images=rendered[1:],
                             duration=durations,loop=0,disposal=2,optimize=False)
            temporary.replace(path)
        finally:
            plt.close(fig);palette.close()
            for frame in rendered:frame.close()
    write_json(archive.with_suffix('.json'),{'case':case,'time_ps':times,'populations':populations,
        'source_snapshots':sources,'archive':str(archive.relative_to(ROOT)),
        'density_scale':{'minimum':0.,'maximum':peak,'units':'um^-2','shared_between_components':True},
        'window_um':[-20,20],'fps':fps,'duration_ms':durations,
        'sampling':'Saved snapshots only; irregular physical times; no interpolation or integration.',
        'style':'Previous polariton-condensate GIF: viridis, fixed palette, 935x451 px, endpoint pauses.'})
    return path


def make_total_density_gif(output_dir=None,*,duration_s=10.,refresh=False):
    """Animate all saved total-density maps with an exact GIF cycle duration."""
    from PIL import Image
    import hashlib
    from numerics import write_json

    if not np.isfinite(duration_s) or duration_s<=0:
        raise ValueError('Animation duration must be positive and finite')
    archive=DATA/'N3_eps_03_total_animation.npz'
    if refresh or not archive.exists():
        source=ROOT/'results/runs/N3_eps_03/core_density.npz'
        with np.load(source,allow_pickle=False) as z:
            ix=np.flatnonzero(abs(z['x'])<=20);iy=np.flatnonzero(abs(z['y'])<=20)
            density=z['density'][:,iy[:,None],ix].copy()
            times=z['time'].copy();x=z['x'][ix].copy();y=z['y'][iy].copy()
        provenance={'path':str(source.relative_to(ROOT)),
                    'sha256':hashlib.sha256(source.read_bytes()).hexdigest()}
        DATA.mkdir(parents=True,exist_ok=True);temporary=archive.with_suffix('.tmp.npz')
        np.savez_compressed(temporary,time=times,density=density,x=x,y=y,
                            source_json=json.dumps(provenance))
        temporary.replace(archive)
    with np.load(archive,allow_pickle=False) as z:
        times=z['time'];density=z['density'];x=z['x'];y=z['y']
        provenance=json.loads(str(z['source_json']))
    if len(times)<2 or np.any(np.diff(times)<=0) or not np.allclose(np.diff(times),np.diff(times)[0]):
        raise ValueError('Uniformly spaced saved density maps are required')
    peak=float(density.max())
    if not np.isfinite(density).all() or peak<=0:raise ValueError('Invalid animation densities')
    total_cs=round(100*duration_s)
    if total_cs<len(times):raise ValueError('Duration is too short for all saved frames')
    # GIF stores centiseconds; distribute rounding without changing total duration.
    boundaries=np.rint(np.linspace(0,total_cs,len(times)+1)).astype(int)
    durations=(10*np.diff(boundaries)).tolist()
    out=ROOT/'results/figures/report' if output_dir is None else Path(output_dir)
    out.mkdir(parents=True,exist_ok=True)
    palette=Image.new('P',(1,1))
    colors=np.vstack((plt.get_cmap('viridis')(np.linspace(0,1,224))[:,:3],
                      np.repeat(np.linspace(0,1,32)[:,None],3,axis=1)))
    palette.putpalette(np.round(255*colors).astype('uint8').ravel().tolist())
    dx=x[1]-x[0];dy=y[1]-y[0]
    extent=(x[0]-dx/2,x[-1]+dx/2,y[0]-dy/2,y[-1]+dy/2)
    rendered=[]
    with plt.rc_context(STYLE):
        fig,ax=plt.subplots(figsize=(5.5,4.6),dpi=110,layout='constrained')
        im=ax.imshow(density[0],origin='lower',extent=extent,cmap='viridis',
                     vmin=0,vmax=peak,interpolation='nearest')
        ax.grid(False)
        ax.set(xlim=(-20,20),ylim=(-20,20),aspect='equal',
               xlabel=r'$x$ ($\mu$m)',ylabel=r'$y$ ($\mu$m)')
        ax.set_title(r'Total condensate density $n_+ + n_-$',fontsize=11)
        ax.set_xticks([-20,0,20]);ax.set_yticks([-20,0,20])
        cb=fig.colorbar(im,ax=ax,shrink=.85,pad=.03)
        cb.set_label(r'$n_+ + n_-$ ($\mu$m$^{-2}$)')
        try:
            for j,t in enumerate(times):
                im.set_data(density[j])
                fig.suptitle(rf'Ring pump $\cdot$ $\epsilon=0.3$ $\cdot$ $t={t:g}$ ps',fontsize=13)
                _layout_check(fig,'N3 total-density animation')
                frame=Image.fromarray(np.asarray(fig.canvas.buffer_rgba())[:,:,:3])
                rendered.append(frame.quantize(palette=palette,dither=Image.Dither.NONE))
                frame.close()
            path=out/'N3_eps_03_total_evolution.gif';temporary=path.with_suffix('.tmp.gif')
            rendered[0].save(temporary,save_all=True,append_images=rendered[1:],
                             duration=durations,loop=0,disposal=2,optimize=False)
            temporary.replace(path)
        finally:
            plt.close(fig);palette.close()
            for frame in rendered:frame.close()
    write_json(archive.with_suffix('.json'),{'case':'N3_eps_03','quantity':'n_plus + n_minus',
        'time_ps':times,'source':provenance,'archive':str(archive.relative_to(ROOT)),
        'duration_ms':durations,'cycle_duration_ms':sum(durations),'endpoint_pauses':False,
        'density_scale':{'minimum':0.,'maximum':peak,'units':'um^-2'},'window_um':[-20,20],
        'sampling':'All uniformly spaced saved total-density maps; no interpolation or integration.',
        'style':'viridis; fixed palette and density scale; 605x506 px; physical time labels.'})
    return path


def report_figure1(out):
    fig,axes=plt.subplots(1,2,figsize=(WIDTH,74/25.4),layout='constrained',sharey=True)
    handles=[];peak=0.
    for name,label in (('M0','One reservoir'),('R1','Two reservoirs')):
        data=load_0d(DATA/f'F1_{name}.npz');obs=observables_0d(data);t=data['time']
        style={'color':COLORS[name],'ls':SCALAR_STYLES[name]}
        handles.append(Line2D([],[],label=label,**style))
        axes[0].plot(t,obs['nc'][:,0],**style)
        axes[1].plot(t-1000,obs['nc'][:,0],**style)
        p=parameters(data['config']);peak=max(peak,float(obs['nc'][t<=350,0].max()))
    steady=.5*p['Pth']/p['gammaC']
    for ax,title in zip(axes,('(a) Pump on','(b) Pump off')):
        ax.axhline(steady,color='.55',ls=':',lw=.9,zorder=0)
        ax.set_title(title,loc='left');ax.set_ylim(-.025,1.06*peak)
        ax.set_ylabel(r'$n_C$ ($\mu$m$^{-2}$)')
    axes[0].set(xlim=(0,350),xlabel='Time after pump on (ps)')
    axes[1].set(xlim=(0,100),xlabel='Time after pump off (ps)')
    fig.legend(handles=handles,loc='outside upper center',ncol=2,frameon=False,
               title=r'Matched models; $P_A=1.5P_{\rm th}$',title_fontsize=8.5)
    return save(fig,out,'N1_reservoir_switching')


def report_figure2(out):
    records=json.loads((DATA/'F3_scan.json').read_text());eps=np.array([r['epsilon'] for r in records])
    n=np.array([r['density'] for r in records]);p=parameters(load_0d(DATA/'F3_eps_+0.0.npz')['config'])
    fine=np.linspace(-1,1,201);analytic=np.array([spin_stationary(2,e,p)[0] for e in fine])
    fig,ax=plt.subplots(figsize=(WIDTH,74/25.4),layout='constrained')
    for k,label in enumerate((r'$+$ circular',r'$-$ circular')):
        ax.plot(fine,analytic[:,k],color=SPIN_COLORS[k],label=label)
        ax.plot(eps,n[:,k],'o',ms=4,mfc='white',mew=1,color=SPIN_COLORS[k])
    ax.set(xlabel=r'Pump polarization, $\epsilon$',ylabel=r'$n_\pm$ ($\mu$m$^{-2}$)',xlim=(-1.04,1.04),ylim=(0,4.4))
    ax.set_xticks([-1,-.5,0,.5,1])
    handles=[*ax.get_legend_handles_labels()[0],
             Line2D([],[],color='.3',label='Steady-state formula'),
             Line2D([],[],ls='',marker='o',mfc='white',color='.3',label='Time integration',ms=4)]
    fig.legend(handles=handles,loc='outside upper center',ncol=2,frameon=False,
               title=r'No coherent coupling; total pump $P_{A,\mathrm{tot}}=4P_{\rm th}$',title_fontsize=8.5)
    return save(fig,out,'N2_circular_populations')


def _report_josephson_cases():
    rows=json.loads((DATA/'F5_scan.json').read_text());cases=[]
    selected=[next(r for r in rows if abs(r['J']-J)<1e-10) for J in (.005,.0065,.010)]
    duration=np.ceil(3*max(r.get('period_ps',0) for r in selected)/.5)*.5
    for row in selected:
        data=load_0d(DATA/f"F5_J_{row['J']:.5f}.npz");obs=observables_0d(data)
        end=row['window_ps'][1];begin=end-duration
        if begin<row['window_ps'][0]:raise ValueError('Selected display window exceeds validated interval')
        mask=(data['time']>=begin-1e-8)&(data['time']<=end+1e-8)
        t=data['time'][mask]
        if np.max(np.diff(t))>.500001:raise ValueError('Example trajectory lacks full time resolution')
        phase=obs['theta'][mask]
        if not np.all(np.isfinite(phase)):raise ValueError('Relative phase is not defined throughout example')
        phase=phase-2*np.pi*np.floor((phase[0]+np.pi)/(2*np.pi))
        cases.append({'J':row['J'],'start_ps':float(begin),'end_ps':float(end),'tau':t-begin,
                      'density':obs['nc'][mask],'phase':phase,'current_mean':float(np.mean(obs['IJ'][mask])),
                      'status':row['status']})
    return cases


def report_figures45(out):
    cases=_report_josephson_cases();paths=[]
    regimes=('Running phase','Oscillating phase','Fixed phase')
    for kind,name in (('density','N4_component_dynamics'),('phase','N5_relative_phase')):
        fig,axes=plt.subplots(1,3,figsize=(WIDTH,82/25.4),layout='constrained')
        for k,(ax,case,regime) in enumerate(zip(axes,cases,regimes)):
            if kind=='density':
                for spin in range(2):ax.plot(case['tau'],case['density'][:,spin],color=SPIN_COLORS[spin])
                ax.set_ylim(0,4.5);ax.set_yticks([0,2,4])
                if k==0:ax.set_ylabel(r'$n_\pm$ ($\mu$m$^{-2}$)')
                else:ax.tick_params(labelleft=False)
            else:
                ax.plot(case['tau'],case['phase'],color='C0')
                ax.set_ylabel(r'$\theta$ (rad)')
                ax.axhline(0,color='.6',lw=.7,zorder=0)
            ax.set_title(f'({chr(97+k)}) {regime}\n'+rf'$J={1000*case["J"]:g}$ $\mu$eV',loc='left',fontsize=9)
            ax.set(xlim=(0,case['end_ps']-case['start_ps']),xlabel=r'$t-t_0$ (ps)')
            ax.set_xticks([0,500,1000])
        if kind=='density':
            handles=[Line2D([],[],color=SPIN_COLORS[k],label=label) for k,label in enumerate((r'$+$ circular',r'$-$ circular'))]
            fig.legend(handles=handles,loc='outside upper center',ncol=2,frameon=False,
                       title=r'$P_{A,\mathrm{tot}}=4P_{\rm th}$; $\epsilon=0.3$',title_fontsize=8.5)
        else:fig.suptitle('Relative phase (different vertical scales); matched time windows',fontsize=8.5)
        paths.append(save(fig,out,name))
    write={k:{field:c[field] for field in ('J','start_ps','end_ps','current_mean','status')}
           for k,c in zip(('J5','J6.5','J10'),cases)}
    from numerics import write_json
    write_json(DATA/'report_windows.json',write)
    return paths


def report_figure6(out):
    rows=sorted((r for r in json.loads((DATA/'F5_scan.json').read_text()) if r['J']<=.010000001),key=lambda r:r['J'])
    fig,axes=plt.subplots(1,2,figsize=(WIDTH,78/25.4),layout='constrained')
    x=np.array([1000*r['J'] for r in rows]);amplitude=np.array([r['A_z'] for r in rows])
    frequency=np.array([r['mean_phase_drift'] if r.get('mean_phase_drift') is not None else np.nan for r in rows])
    for ax,values in zip(axes,(amplitude,frequency)):
        ax.plot(x,values,'o-',color='C0',ms=3,mfc='white',mew=.8,lw=.8)
        for j,marker,color in ((5.,'o','C1'),(6.5,'s','C2'),(10.,'D','C3')):
            k=int(np.argmin(abs(x-j)))
            if np.isfinite(values[k]):ax.plot(x[k],values[k],marker,ms=6,color=color,zorder=5)
        ax.set(xlim=(-.15,10.3),xlabel=r'Coherent coupling, $J$ ($\mu$eV)')
        ax.set_xticks([0,2,4,6,8,10])
    masked=np.array([r['status']=='PERIODIC_PHASE_MASKED' for r in rows])
    axes[0].plot(x[masked],amplitude[masked],'D',color='.45',ms=3.7,zorder=4)
    axes[0].set_title('(a) Polarization oscillations',loc='left');axes[0].set_ylabel('Amplitude, $A_z$')
    axes[0].set_ylim(-.025,max(.55,1.08*float(amplitude.max())));axes[0].set_yticks([0,.2,.4,.6])
    axes[0].annotate(r'$6.84$-$6.86$ $\mu$eV',xy=(6.85,.005),xytext=(8.55,.32),
                     ha='center',fontsize=8,color='.25',
                     arrowprops={'arrowstyle':'->','lw':.8,'color':'.35'})
    axes[1].set_title('(b) Mean phase drift',loc='left')
    axes[1].set_ylabel(r'$\langle\dot\theta\rangle$ (rad ps$^{-1}$)')
    axes[1].ticklabel_format(axis='y',style='sci',scilimits=(-2,-2),useMathText=True)
    axes[1].axhline(0,color='.6',lw=.7,zorder=0)
    handles=[Line2D([],[],ls='',marker=m,color=c,label=rf'$J={j:g}$ $\mu$eV',ms=5)
             for j,m,c in ((5,'o','C1'),(6.5,'s','C2'),(10,'D','C3'))]
    fig.legend(handles=handles,loc='outside upper center',ncol=3,frameon=False,
               title='Selected homogeneous regimes',title_fontsize=8.5)
    return save(fig,out,'N6_coupling_summary')


def report_figure3(out):
    import matplotlib.patheffects as effects
    report=json.loads((DATA/'N3_summary.json').read_text())
    if report['status']!='COMPLETED':raise ValueError('N3 calculation is still in progress')
    cases=[]
    with np.load(DATA/'spatial_spin.npz',allow_pickle=False) as z:
        for name in ('N3_eps_0','N3_eps_03'):
            c=json.loads(str(z[name+'_config']));psi=z[name+'_psi'].copy()
            cases.append((c,abs(psi)**2,report['cases'][name]))
    times=[s['time_ps'] for _,_,s in cases]
    if times[0]!=times[1]:raise ValueError('N3 needs a common final time')
    peak=max(float(n.max()) for _,n,_ in cases)
    fig=plt.figure(figsize=(WIDTH,140/25.4),layout='compressed')
    gs=fig.add_gridspec(2,3,width_ratios=(1,1,.055))
    for row,(c,n,summary) in enumerate(cases):
        g=make_grid(c);extent=(g.x[0]-g.dx/2,g.x[-1]+g.dx/2,g.y[0]-g.dx/2,g.y[-1]+g.dx/2)
        for spin in range(2):
            ax=fig.add_subplot(gs[row,spin]);ax.grid(False)
            im=ax.imshow(n[spin],origin='lower',extent=extent,cmap='magma',vmin=0,vmax=peak,interpolation='nearest')
            line=ax.contour(g.x,g.y,pump_profile(c,g),levels=[.5],colors='white',linestyles='--',linewidths=.65)
            line.set_path_effects([effects.Stroke(linewidth=1.1,foreground='.1'),effects.Normal()])
            ax.set(xlim=(-20,20),ylim=(-20,20),aspect='equal')
            ax.set_xticks([-20,0,20]);ax.set_yticks([-20,0,20])
            label='+' if spin==0 else '-'
            ax.set_title(f'({chr(97+2*row+spin)}) '+rf'$n_{label}$; $\epsilon={c["pump"]["epsilon"]:g}$',loc='left',fontsize=9)
            ax.text(.97,.95,rf'$N_{label}={summary["component_populations"]["NC"][spin]:.2f}$',
                    transform=ax.transAxes,ha='right',va='top',color='white',fontsize=8)
            if row==1:ax.set_xlabel(r'$x$ ($\mu$m)')
            else:ax.tick_params(labelbottom=False)
            if spin==0:ax.set_ylabel(r'$y$ ($\mu$m)')
            else:ax.tick_params(labelleft=False)
    cax=fig.add_subplot(gs[:,2]);cax.grid(False);fig.colorbar(im,cax=cax,label=r'$n_\pm$ ($\mu$m$^{-2}$)')
    states='Steady densities' if all(s['two_windows_pass'] for _,_,s in cases) else 'Density snapshots'
    fig.suptitle(f'{states} at $t={times[0]:g}$ ps; no coherent coupling',fontsize=9)
    return save(fig,out,'N3_spatial_components')


def make_report_figures(*,output_dir=None):
    """Redraw the simpler N figures; never overwrite existing F/D figures."""
    out=ROOT/'results/figures/report' if output_dir is None else Path(output_dir)
    with plt.rc_context(STYLE):
        paths=[report_figure1(out),report_figure2(out),*report_figures45(out),report_figure6(out)]
        summary=DATA/'N3_summary.json'
        if summary.exists() and json.loads(summary.read_text()).get('status')=='COMPLETED':
            paths.append(report_figure3(out))
            diagnostic_out=ROOT/'results/diagnostics' if output_dir is None else out/'diagnostics'
            paths.append(report_spatial_diagnostic(diagnostic_out))
            if output_dir is None:_update_spatial_notes(json.loads(summary.read_text()))
    return sorted(paths)


def report_spatial_diagnostic(out):
    """Individual populations and masked local circular polarization for N3."""
    from matplotlib.colors import Normalize
    fig,axes=plt.subplots(2,2,figsize=(WIDTH,130/25.4),layout='constrained',sharey='row')
    peak_population=0.
    with np.load(DATA/'spatial_spin.npz',allow_pickle=False) as z:
        for col,name in enumerate(('N3_eps_0','N3_eps_03')):
            c=json.loads(str(z[name+'_config']));g=make_grid(c);h=z[name+'_history'];n=abs(z[name+'_psi'])**2
            for k,label in enumerate((r'$N_+$',r'$N_-$')):
                axes[0,col].plot(h[:,0],h[:,6+k],color=SPIN_COLORS[k],ls='-' if k==0 else '--',label=label)
            peak_population=max(peak_population,float(h[:,6:8].max()))
            axes[0,col].axvspan(h[-1,0]-200,h[-1,0],color='.5',alpha=.1,zorder=0)
            axes[0,col].set(xlabel='Time (ps)',ylabel='Condensate population')
            axes[0,col].set_xlim(0,h[-1,0])
            axes[0,col].set_title(f'({chr(97+col)}) '+rf'$\epsilon={c["pump"]["epsilon"]:g}$',loc='left')
            axes[0,col].legend(frameon=False)
            total=n.sum(axis=0);valid=total>max(1e-8,1e-4*float(total.max()))
            polarization=np.divide(n[0]-n[1],total,out=np.full_like(total,np.nan),where=valid)
            extent=(g.x[0]-g.dx/2,g.x[-1]+g.dx/2,g.y[0]-g.dx/2,g.y[-1]+g.dx/2)
            im=axes[1,col].imshow(polarization,origin='lower',extent=extent,cmap='coolwarm',norm=Normalize(-1,1),interpolation='nearest')
            axes[1,col].set(xlim=(-20,20),ylim=(-20,20),aspect='equal',xlabel=r'$x$ ($\mu$m)',ylabel=r'$y$ ($\mu$m)')
            axes[1,col].set_title(f'({chr(99+col)}) Local polarization',loc='left');axes[1,col].grid(False)
    axes[0,0].set_ylim(0,1.05*peak_population)
    fig.colorbar(im,ax=axes[1,:].tolist(),label=r'$z(x,y)$',shrink=.85,ticks=[-1,0,1])
    return save(fig,out,'D4_spatial_spin_diagnostics')


def _update_spatial_notes(report):
    """Refresh only the marked N3 result blocks after the saved calculation."""
    import re
    a,b=(report['cases'][k] for k in ('N3_eps_0','N3_eps_03'))
    overview=["### N3. Пространственные круговые компоненты\n",
              "[Рисунок N3](../results/figures/report/N3_spatial_components.png).\n",
              "**English caption — Spatial circular components under ring pumping.** "
              "Columns show the two circular-component densities; rows compare pump polarizations "
              f"$\\epsilon=0$ and $0.3$ at the same total pump and $t={a['time_ps']:g}$ ps. All four maps share one "
              "absolute color scale. Dashed contours mark half-maximum pump intensity. "
              "The labels give integrated populations, not peak-normalized densities. "
              "This is an inhomogeneous calculation without coherent coupling ($J=0$).\n",
              "| ε | t, пс | N₊ | N₋ | Z | Два окна установления |\n|---|---:|---:|---:|---:|---|\n"]
    for eps,s in ((0,a),(.3,b)):
        n=s['component_populations']['NC'];status='PASS' if s['two_windows_pass'] else 'UNRESOLVED'
        overview.append(f"| {eps:g} | {s['time_ps']:g} | {n[0]:.6f} | {n[1]:.6f} | {s['Z']:.6f} | {status} |\n")
    with np.load(DATA/'spatial_spin.npz',allow_pickle=False) as z:
        n=abs(z['N3_eps_03_psi'])**2
    normalized=n/n.sum(axis=(-2,-1))[:,None,None]
    difference=float(np.linalg.norm(normalized[0]-normalized[1])/np.linalg.norm(normalized[0]))
    overview.append(f"\nПри ε=0.3 основной видимый эффект — различие населённостей (Z={b['Z']:.5f}); нормированные пространственные профили остаются близкими. "
                    f"Их относительное L2-различие составляет {100*difference:.3f}%. "
                    "Для $\\widehat n_\\sigma=n_\\sigma/N_\\sigma$ используется "
                    "$\\delta_{\\rm shape}=\\|\\widehat n_+-\\widehat n_-\\|_2/\\|\\widehat n_+\\|_2$ "
                    "по всем узлам области 256×256 мкм, без поворота и сдвига; в знаменателе — профиль компоненты +. "
                    "Симметричный опыт проверяет равенство компонент при симметричной подготовке; "
                    "он не доказывает выбор этого состояния из любых начальных условий.\n")
    if not all(s['two_windows_pass'] for s in (a,b)):
        overview.append("\n**Плотности не прошли проверку установления: карты являются снимками, а не стационарными решениями.**\n")
    overview.append("\nЛокальная поляризация и истории N₊/N₋: [D4](../results/diagnostics/D4_spatial_spin_diagnostics.png); "
                    "у историй общая вертикальная шкала, серым выделены последние два окна по 100 пс. "
                    "Белые области карт поляризации маскируют недостаточную суммарную плотность. "
                    "Поля, резервуары, конфигурации и истории: `results/data/spatial_spin.npz`; "
                    "численные проверки: `N3_summary.json`.\n")
    diagnostic=["## N3: пространственная спиновая серия\n",
                f"Итоговая оценка: **{report['assessment']}**. Все ошибки ниже — относительные доли.\n",
                "| Расчёт | t, пс | Время, мин | Стационарность обеих компонент | Баланс | Край / high-k, максимум компонент | Статус |\n"
                "|---|---:|---:|---|---:|---|---|\n"]
    for name,s in report['cases'].items():
        edge=max(v['edge_mass'] for v in s['spatial_components']);high=max(v['high_k'] for v in s['spatial_components'])
        diagnostic.append(f"| {name} | {s['time_ps']:g} | {s['elapsed_wall_s']/60:.2f} | "
                          f"{'PASS' if s['two_windows_pass'] else 'UNRESOLVED'} | {s['balance']['relative']:.3g} | {edge:.3g} / {high:.3g} | {s['checks']} |\n")
    diagnostic.append("\n| Сравнение | ΔN₊ | ΔN₋ | L2(n₊) | L2(n₋) | Статус |\n|---|---:|---:|---:|---:|---|\n")
    for name,r in report['comparisons'].items():
        diagnostic.append(f"| {name} | {r['NC_relative'][0]:.3g} | {r['NC_relative'][1]:.3g} | "
                          f"{r['density_L2_relative'][0]:.3g} | {r['density_L2_relative'][1]:.3g} | {r['status']} |\n")
    diagnostic.append("\nДопуск сходимости: 1% для каждой населённости и каждой абсолютной карты. "
                      "На вложенных сетках сравниваются совпадающие физические узлы без подгонки амплитуды и поворота. "
                      "Ошибки карт равны $\\|n^{(c)}-\\mathcal S n^{(\\rm ref)}\\|_2/\\|\\mathcal S n^{(\\rm ref)}\\|_2$, "
                      "где эталон — основной расчёт, а S выбирает его значения в узлах контрольной сетки. "
                      "Стационарность проверена отдельно по N±, обеим плотностям и всем четырём резервуарам "
                      "на двух последовательных окнах по 100 пс: относительные размах и дрейф между средними по половинам окна <0.5%, "
                      "относительные L2-изменения карт от начала окна <1%; постоянство относительной фазы не требуется.\n")
    for filename,tag,content in (('results_overview.md','N3_RESULT','\n'.join(overview)),
                                  ('diagnostic_summary.md','N3_CHECKS','\n'.join(diagnostic))):
        content=re.sub(r'(?m)(^\|.*\|)\n(?:[ \t]*\n)+(?=\|)',r'\1\n',content)
        path=ROOT/'docs/journal'/filename;text=path.read_text();start=f'<!-- {tag}_START -->';end=f'<!-- {tag}_END -->'
        if start not in text or end not in text:raise ValueError('Missing documentation markers: '+filename)
        before,rest=text.split(start,1);_,after=rest.split(end,1)
        path.write_text(before+start+'\n'+content+'\n'+end+after)
    _update_latex_spatial_results(report)


def _update_latex_spatial_results(report):
    """Update both report languages from the same measured N3 results."""
    import re,shutil,subprocess
    path=ROOT/'latex/report_ru.tex'
    if not path.exists():return
    cases=[report['cases'][name] for name in ('N3_eps_0','N3_eps_03')]
    t=cases[0]['time_ps']
    result=[rf'Карты на рис.~\ref{{fig:3}} соответствуют общему времени $t={t:g}$ пс. Интегральные величины вычислены по всей области $256\times256$ мкм.',
            r'\begin{center}\begin{tabular}{@{}crrrc@{}}\toprule',
            r'$\varepsilon$ & $N_+$ & $N_-$ & $Z$ & Установление\\\midrule']
    for eps,s in zip((0,.3),cases):
        n=s['component_populations']['NC'];state='да' if s['two_windows_pass'] else 'нет'
        result.append(f'{eps:g} & {n[0]:.4f} & {n[1]:.4f} & {s["Z"]:.5f} & {state}'+r'\\')
    result.append(r'\bottomrule\end{tabular}\end{center}')
    if all(s['two_windows_pass'] for s in cases):
        result.append('Обе компоненты и соответствующие резервуары прошли проверку установления на двух последовательных окнах по 100 пс.')
    else:result.append(r'\textbf{Не все плотности прошли проверку установления.} Соответствующие карты являются снимками в указанный момент, а не стационарными решениями.')
    if report['symmetric_preparation']['status']=='PASS':
        result.append('При симметричной накачке и одинаковых начальных полях плотности компонент совпадают. Это контроль симметричной подготовки, а не доказательство выбора равных плотностей из любых начальных условий.')
    with np.load(DATA/'spatial_spin.npz',allow_pickle=False) as z:
        n=abs(z['N3_eps_03_psi'])**2
    totals=np.sum(n,axis=(-2,-1))
    if min(cases[1]['component_populations']['NC'])>1e-3:
        difference=float(np.linalg.norm(n[0]/totals[0]-n[1]/totals[1])/np.linalg.norm(n[0]/totals[0]))
        profile_metric=(r'\[\begin{gathered}'
            r'\widehat n_\sigma=\frac{n_\sigma}{N_\sigma},\qquad '
            r'\|f\|_{2,h}=\left(\Delta x\,\Delta y\sum_{i,j}|f_{ij}|^2\right)^{1/2},\\[0.4em]'
            r'\delta_{\rm shape}=\frac{\|\widehat n_+-\widehat n_-\|_{2,h}}{\|\widehat n_+\|_{2,h}}='
            +f'{100*difference:.3g}'+r'\%.\end{gathered}\]')
        result.append(r'При $\varepsilon=0.3$ дисбаланс накачки прежде всего меняет населённости компонент: $Z='+f'{cases[1]["Z"]:.5f}'+r'$. Пространственные профили остаются близкими, но не совпадают. Их относительное различие после нормировки каждого на соответствующую населённость определяется как')
        result.append(profile_metric)
        result.append(r'Норма вычисляется по всем узлам области $256\times256$ мкм без сдвига и поворота. В знаменателе стоит норма нормированного профиля компоненты $+$. На рис.~\ref{fig:3} показаны абсолютные плотности; нормировка используется только для сравнения формы.')
    else:result.append('Одна из компонент слишком слаба для надёжного нормированного сравнения форм.')
    def number(value):
        if value==0:return '$0$'
        mantissa,exponent=f'{value:.2e}'.split('e')
        return '$'+mantissa+r'\times10^{'+str(int(exponent))+'}$'
    error_metric=(r'\[\begin{gathered}'
        r'\delta N_\sigma=\frac{|N_\sigma^{(c)}-N_\sigma^{(\rm ref)}|}{|N_\sigma^{(\rm ref)}|},\\[0.4em]'
        r'\delta_2(n_\sigma)=\frac{\|n_\sigma^{(c)}-\mathcal S n_\sigma^{(\rm ref)}\|_{2,h_c}}'
        r'{\|\mathcal S n_\sigma^{(\rm ref)}\|_{2,h_c}}.\end{gathered}\]')
    checks=[r'Для $\varepsilon=0.3$ выполнены независимые повторы с половинным временным шагом и грубой пространственной сеткой. Относительные ошибки определяются как',
            error_metric,
            r'Здесь ref --- основной расчёт на $512^2$ с шагом $0.025$ пс, $c$ --- контроль. Оператор $\mathcal S$ выбирает узлы контрольной сетки из эталона и тождественен при сравнении шагов. Норма охватывает всю область на сетке контроля. В таблице приведены $100\delta N_\sigma$ и $100\delta_2(n_\sigma)$ при общем конечном времени, то есть ошибки в процентах.',
            r'\begin{center}\small\begin{tabular}{@{}lrrrrc@{}}\toprule',
            r'Контроль & $\delta N_+$ & $\delta N_-$ & $\delta_2(n_+)$ & $\delta_2(n_-)$ & Статус\\\midrule']
    for key,label in (('grid256',r'Сетка $256^2$'),('dt_half',r'Шаг $0.0125$ пс')):
        row=report['comparisons'][key];values=[*row['NC_relative'],*row['density_L2_relative']]
        checks.append(label+' & '+' & '.join(number(100*v) for v in values)+' & '+row['status']+r'\\')
    checks.append(r'\bottomrule\end{tabular}\end{center}')
    checks.append(r'Обе ошибки должны быть менее $1\%$ для каждой компоненты. Поворот и подгонка амплитуды не применяются.')
    balance=max(s['balance']['relative'] for s in report['cases'].values())
    edge=max(v['edge_mass'] for s in report['cases'].values() for v in s['spatial_components'])
    high=max(v['high_k'] for s in report['cases'].values() for v in s['spatial_components'])
    checks.append('Максимальные относительные значения по четырём расчётам: невязка общего баланса '+number(balance)+', краевой вес '+number(edge)+', высокочастотный вес '+number(high)+'. Пороговые значения: '+r'$10^{-3}$, $10^{-4}$ и $10^{-6}$ соответственно.')
    checks.append('Итоговая оценка новой пространственной серии: '+r'\textbf{'+report['assessment'].replace('_',r'\_')+'}.')
    status='' if report['assessment']=='PASS' else r'\begin{center}\small\textbf{Пространственная серия завершена; часть численных проверок требует анализа.}\end{center}'
    result_en=[rf'The maps in Fig.~\ref{{fig:3}} correspond to the common final time $t={t:g}$ ps. Populations are integrated over the full $256\times256$ $\mu\mathrm m$ domain.',
               r'\begin{center}\begin{tabular}{@{}crrrc@{}}\toprule',
               r'$\varepsilon$ & $N_+$ & $N_-$ & $Z$ & Stationarity\\\midrule']
    for eps,s in zip((0,.3),cases):
        n=s['component_populations']['NC'];state='yes' if s['two_windows_pass'] else 'no'
        result_en.append(f'{eps:g} & {n[0]:.4f} & {n[1]:.4f} & {s["Z"]:.5f} & {state}'+r'\\')
    result_en.append(r'\bottomrule\end{tabular}\end{center}')
    if all(s['two_windows_pass'] for s in cases):
        result_en.append('Both condensate components and their reservoirs pass the stationarity criteria over two consecutive 100 ps windows.')
    else:
        result_en.append(r'\textbf{Not all densities pass the stationarity criteria.} The corresponding maps are snapshots at the stated time, rather than stationary solutions.')
    if report['symmetric_preparation']['status']=='PASS':
        result_en.append('For symmetric pumping and equal initial fields, the component densities coincide. This checks the symmetric preparation; it does not establish convergence to equal densities from arbitrary initial conditions.')
    if min(cases[1]['component_populations']['NC'])>1e-3:
        result_en.append(r'At $\varepsilon=0.3$, the pump imbalance mainly changes the component populations, giving $Z='+f'{cases[1]["Z"]:.5f}'+r'$. The spatial profiles remain similar, but do not coincide. Their relative difference after normalization by their respective populations is defined as')
        result_en.append(profile_metric)
        result_en.append(r'The norm uses all grid nodes in the $256\times256$ $\mu\mathrm m$ domain, without translation or rotation. The denominator is the norm of the normalized $+$ component profile. The maps in Fig.~\ref{fig:3} show absolute densities; normalization is used only to compare profile shapes.')
    else:
        result_en.append('One component is too weak for a reliable comparison of normalized profile shapes.')
    checks_en=[r'At $\varepsilon=0.3$, independent runs use a halved time step and a coarser spatial grid. Relative errors are defined as',
               error_metric,
               r'Here ref is the main $512^2$ run with a $0.025$ ps step, and $c$ is the check. The operator $\mathcal S$ samples the reference at check-grid nodes and is the identity for the time-step comparison. The norm covers the full domain on the check grid. The table lists $100\delta N_\sigma$ and $100\delta_2(n_\sigma)$ at the common final time, i.e. percent errors.',
               r'\begin{center}\small\begin{tabular}{@{}lrrrrc@{}}\toprule',
               r'Check & $\delta N_+$ & $\delta N_-$ & $\delta_2(n_+)$ & $\delta_2(n_-)$ & Status\\\midrule']
    for key,label in (('grid256',r'$256^2$ grid'),('dt_half',r'$0.0125$ ps step')):
        row=report['comparisons'][key];values=[*row['NC_relative'],*row['density_L2_relative']]
        checks_en.append(label+' & '+' & '.join(number(100*v) for v in values)+' & '+row['status']+r'\\')
    checks_en.append(r'\bottomrule\end{tabular}\end{center}')
    checks_en.append(r'Both errors must be below $1\%$ for each component. No rotation or amplitude fitting is applied.')
    checks_en.append('Maximum relative values over the four runs: total particle-balance residual '+number(balance)+', boundary density fraction '+number(edge)+', and high-wave-number spectral weight '+number(high)+r'. Their respective thresholds are $10^{-3}$, $10^{-4}$, and $10^{-6}$.')
    checks_en.append('Overall assessment of the new spatial series: '+r'\textbf{'+report['assessment'].replace('_',r'\_')+'}.')
    status_en='' if report['assessment']=='PASS' else r'\begin{center}\small\textbf{The spatial series is complete; some numerical checks require further analysis.}\end{center}'
    reports=[(path,{'N3_RESULT':'\n'.join(result),'N3_CHECKS':'\n'.join(checks),'REPORT_STATUS':status}),
             (path.with_name('report_en.tex'),{'N3_RESULT':'\n'.join(result_en),'N3_CHECKS':'\n'.join(checks_en),'REPORT_STATUS':status_en})]
    for path,blocks in reports:
        if not path.exists():continue
        text=path.read_text()
        for tag,content in blocks.items():
            pattern=r'(?m)^% '+tag+r'_START\n.*?^% '+tag+r'_END'
            replacement='% '+tag+'_START\n'+content+'\n% '+tag+'_END'
            text,count=re.subn(pattern,lambda m:replacement,text,flags=re.S)
            if count!=1:raise ValueError('Missing report markers: '+tag+' in '+path.name)
        path.write_text(text)
        if shutil.which('latexmk'):
            built=subprocess.run(['latexmk','-pdf','-quiet','-interaction=nonstopmode','-halt-on-error',path.name],cwd=path.parent,capture_output=True,text=True,errors='replace')
            if built.returncode:raise RuntimeError('Report compilation failed: '+built.stdout[-3000:]+built.stderr[-1000:])
            subprocess.run(['latexmk','-c','-quiet',path.name],cwd=path.parent,capture_output=True,check=True)
            print('Report updated:',path.with_suffix('.pdf'),flush=True)


def make_report3_pilot_figures():
    """First-stage figures only, from saved arrays; earlier figures/reports untouched."""
    directory=DATA/'report3';file=directory/'pilot_summary.json'
    if not file.exists():return []
    report=json.loads(file.read_text());power=report.get('selected_p');captions={};paths=[]
    _report3_notes(report)
    if power is None:return paths
    out=ROOT/'results/figures/report3';diagnostics=ROOT/'results/diagnostics/report3'
    def load(case,geometry):
        name=f'p{power:g}_{case}_{geometry}'
        with np.load(directory/(name+'.npz')) as z:return {k:z[k].copy() for k in z.files}
    def density_caption(data,geometry):
        cfg=json.loads(str(data['config_json']));p=parameters(cfg)
        interval=data['mean_interval_ps'];area=(data['x'][1]-data['x'][0])**2
        pop=data['mean_maps'][1:3].sum((-2,-1))*area
        return (f'{geometry}; p={power:g}, epsilon={cfg["pump"]["epsilon"]:g}, '
                f'alpha2={p["alpha2"]:g} meV um^2, J={p["J"]:g} meV, beta={p["beta"]:g} meV um^2. '
                f'Time means over {interval[0]:g}–{interval[1]:g} ps; N+={pop[0]:.5g}, N-={pop[1]:.5g}, '
                f'Z={(pop[0]-pop[1])/max(pop.sum(),1e-300):.5g}. ')
    def draw(ax,values,data,title,*,vmin=0.,vmax=None,cmap='viridis',mask=False):
        x=data['x'];keep=abs(x)<=40.;image=values[np.ix_(keep,keep)]
        if mask:
            S0=data['mean_maps'][0][np.ix_(keep,keep)]
            image=np.ma.masked_where(S0<1e-3*data['mean_maps'][0].max(),image)
        xcrop=x[keep];dx=x[1]-x[0]
        im=ax.imshow(image,origin='lower',extent=[xcrop[0]-dx/2,xcrop[-1]+dx/2]*2,
                     vmin=vmin,vmax=vmax,cmap=cmap,interpolation='nearest')
        cfg=json.loads(str(data['config_json']));g=make_grid(cfg);pump=pump_profile(cfg,g)
        ax.contour(xcrop,xcrop,pump[np.ix_(keep,keep)],levels=[.5],colors='0.25' if mask else 'white',linewidths=.65)
        ax.set(title=title,xlabel=r'$x$ ($\mu$m)');ax.grid(False);ax.set_aspect('equal')
        return im
    with plt.rc_context(STYLE):
        if all((directory/f'p{power:g}_A1_{geom}.npz').exists() for geom in ('spot','ring')):
            fig,axes=plt.subplots(2,2,figsize=(WIDTH,116/25.4),layout='constrained')
            caption='Circular-component densities with polarized pumping. '
            for row,geom in enumerate(('spot','ring')):
                d=load('A1',geom);vmax=float(d['mean_maps'][1:3].max())
                for col,label in enumerate(('+','-')):
                    im=draw(axes[row,col],d['mean_maps'][1+col],d,
                            f'({chr(97+2*row+col)}) {geom.capitalize()}: '+r'$n_'+label+'$',vmax=vmax)
                axes[row,0].set_ylabel(r'$y$ ($\mu$m)')
                fig.colorbar(im,ax=axes[row],label=r'Mean density ($\mu$m$^{-2}$)',fraction=.04,pad=.02)
                caption+=density_caption(d,geom)
            name='pilot_A1_circular_densities';paths.append(save(fig,out,name));captions[name]=caption+'Gaussian width w=2.5 um; ring radius r0=10 um and width d=2.5 um. Both components of each geometry share one absolute density scale. White contours mark half-maximum pump. Maps are cropped to +/-40 um; populations use the full 256 um box.'
        for geom in ('spot','ring'):
            if not (directory/f'p{power:g}_T0_{geom}.npz').exists():continue
            d=load('T0',geom)
            if json.loads(str(d['summary_json']))['late_NC_to_seed']<100:
                captions['pilot_T0_'+geom]='No polarization map: condensate population is below the operational nonzero-signal criterion.'
                (out/('pilot_T0_'+geom+'.png')).unlink(missing_ok=True)
                continue
            maps=d['mean_maps'];s3=np.divide(maps[9],maps[0],out=np.zeros_like(maps[0]),where=maps[0]>0)
            fig,axes=plt.subplots(1,3,figsize=(WIDTH,66/25.4),layout='constrained')
            for col,label in enumerate(('+','-')):
                im=draw(axes[col],maps[1+col],d,f'({chr(97+col)}) '+r'Mean $n_'+label+'$',vmax=float(maps[1:3].max()))
            fig.colorbar(im,ax=axes[:2],label=r'Density ($\mu$m$^{-2}$)',fraction=.035,pad=.02)
            im=draw(axes[2],s3,d,r'(c) Circular $s_3$',vmin=-1,vmax=1,cmap='RdBu_r',mask=True)
            fig.colorbar(im,ax=axes[2],ticks=[-1,0,1],fraction=.06,pad=.025)
            axes[0].set_ylabel(r'$y$ ($\mu$m)');fig.suptitle('Gaussian spot' if geom=='spot' else 'Annular pump',fontsize=10)
            name='pilot_T0_'+geom;paths.append(save(fig,out,name))
            captions[name]=density_caption(d,geom)+'TE–TM coupling under equal circular pump rates, x-linearly-polarized radial initial seed. Gaussian width 2.5 um; ring radius 10 um, width 2.5 um. Normalized polarization is the ratio of time-averaged UNNORMALIZED S3 and S0, not an unweighted mean of local ratios. Pixels below 0.001 of the peak mean S0 are masked; thin contours mark half-maximum pump. No claim of a stationary state is made solely from these mean maps.'
        for geom in ('spot','ring'):
            if not all((directory/f'p{power:g}_{case}_{geom}.npz').exists() for case in ('B0','A1','T0')):continue
            fig,axes=plt.subplots(2,3,figsize=(WIDTH,105/25.4),layout='constrained',sharex=True)
            for j,case in enumerate(('B0','A1','T0')):
                d=load(case,geom);h=d['history'];f=d['features'];row=report['cases'][f'p{power:g}_{case}_{geom}']
                interval=report.get('comparison_intervals',{}).get(geom,row['analysis']['common_interval_ps'])
                for k in (0,1):axes[0,j].plot(h[:,0],h[:,k+1],label=r'$N_'+('+' if k==0 else '-')+'$')
                for k in range(3):axes[1,j].plot(f[:,0],np.where(f[:,1:3].sum(1)>=max(1e-8,100*json.loads(str(d['metadata_json']))['initial_NC']),f[:,7+k]/np.maximum(f[:,1:3].sum(1),1e-300),np.nan),label=r'$\bar s_'+str(k+1)+'$')
                axes[0,j].set_title(f'{case}: '+row['analysis']['status'].lower())
                for ax in axes[:,j]:ax.set_xlim(max(0,interval[0]),interval[1]);ax.ticklabel_format(axis='x',style='plain',useOffset=False)
                axes[1,j].set_xlabel('Time (ps)');axes[1,j].set_ylim(-1.08,1.08)
            axes[0,0].set_ylabel('Population');axes[1,0].set_ylabel('Integrated polarization')
            axes[0,0].legend(frameon=False);axes[1,0].legend(frameon=False,ncol=3,columnspacing=.5,handlelength=.7)
            name='pilot_history_'+geom;paths.append(save(fig,diagnostics,name))
            captions[name]=f'{geom.capitalize()} pilots B0/A1/T0 at p={power:g}. Individual populations and intensity-weighted integrated Stokes parameters over the common late comparison interval. B0: epsilon=0,beta=0; A1: epsilon=0.3,beta=0; T0: epsilon=0,beta=0.0119 meV um^2. J=0 and alpha2=-0.00024 meV um^2 throughout. The labels describe the full-map classification, not a conclusion based on the population traces alone.'
    from numerics import write_json
    write_json(directory/'figure_captions.json',captions)
    return paths


def _report3_notes(report):
    """One short marked addition to the existing results file, no new MD files."""
    path=ROOT/'docs/journal/results_overview.md';text=path.read_text();tag='REPORT3_PILOTS'
    lines=['## Новая серия: предварительная проверка TE–TM',
           f'Статус: **{report["status"]}**. Здесь выполняются только B0/A1/T0; остальные расчёты не запущены.',
           'Независимые радиальные затравки, пустые резервуары; L=256 мкм, 512², dx=0.5 мкм, dt=0.025 пс. Intel Arc, complex128/float64, скомпилированный совместный RK4 и точная FFT-экспонента. Температурных остановок нет.',
           '| Прогон | t, пс | Время, мин | N+ | N− | Режим | Баланс C / всех частиц |',
           '|---|---:|---:|---:|---:|---|---|']
    for name,row in report.get('cases',{}).items():
        phase=row['analysis']['status'] if row['late_NC_to_seed']>=100 else 'NO_CONDENSATE'
        lines.append(f'| {name} | {row["time_ps"]:g} | {row["elapsed_wall_s"]/60:.2f} | {row["NC"][0]:.5g} | {row["NC"][1]:.5g} | {phase} | {row["condensate_balance_relative"]:.2g} / {row["total_balance_relative"]:.2g} |')
    lines+=['','Ненулевая плотность и установление — разные проверки. Угловой показатель Q2=2|C2| не является топологическим зарядом. Карты и поздние интервалы сравниваются отдельно от сходимости по сетке/шагу; такая сходимость новой серии пока не заявлена.',
            'Полные настройки, невязки аналитических тестов и переносимые массивы: `results/data/report3/`. Подробные поля для диагностики и продолжения: игнорируемая `results/runs/report3/`. Рисунки появляются в `results/figures/report3/` после расчёта. Текст прежних отчётов не меняется.']
    block='<!-- '+tag+'_START -->\n'+'\n\n'.join(lines[:3])+'\n\n'+'\n'.join(lines[3:])+'\n<!-- '+tag+'_END -->'
    import re
    pattern='<!-- '+tag+'_START -->.*?<!-- '+tag+'_END -->'
    if re.search(pattern,text,flags=re.S):text=re.sub(pattern,lambda _:block,text,flags=re.S)
    else:text+='\n\n'+block+'\n'
    path.write_text(text)


def make_night_review_figure():
    """Core T0 snapshots; diagnostic only, not a convergence claim."""
    directory=DATA/'report3';out=ROOT/'results/diagnostics/report3'
    with plt.rc_context(STYLE):
        fig,axes=plt.subplots(2,2,figsize=(WIDTH,110/25.4),layout='constrained')
        for row,geometry in enumerate(('spot','ring')):
            with np.load(directory/f'production_T0_{geometry}.npz',allow_pickle=False) as z:
                psi=z['psi'];x=z['x'];cfg=json.loads(str(z['config_json']))
            from model import stokes
            S=stokes(psi);keep=abs(x)<=25.;xx=x[keep];dx=x[1]-x[0]
            n=S[0][np.ix_(keep,keep)]
            s=np.divide(S[3],S[0],out=np.zeros_like(S[0]),where=S[0]>0)[np.ix_(keep,keep)]
            s=np.ma.masked_where(n<1e-3*S[0].max(),s)
            extent=[xx[0]-dx/2,xx[-1]+dx/2]*2
            im=axes[row,0].imshow(n,origin='lower',extent=extent,cmap='viridis',vmin=0,interpolation='nearest')
            fig.colorbar(im,ax=axes[row,0],label=r'$n_++n_-$ ($\mu$m$^{-2}$)',fraction=.045,pad=.02)
            pol=axes[row,1].imshow(s,origin='lower',extent=extent,cmap='RdBu_r',vmin=-1,vmax=1,interpolation='nearest')
            pump=pump_profile(cfg,make_grid(cfg))[np.ix_(keep,keep)]
            for j,ax in enumerate(axes[row]):
                ax.contour(xx,xx,pump,levels=[.5],colors='white' if j==0 else '.25',linewidths=.6)
                ax.set(xlabel=r'$x$ ($\mu$m)',ylabel=r'$y$ ($\mu$m)');ax.grid(False)
            label='Gaussian spot' if geometry=='spot' else 'Annular pump'
            axes[row,0].set_title(f'({chr(97+2*row)}) {label}',loc='left')
            axes[row,1].set_title(f'({chr(98+2*row)}) '+r'Circular $s_3$',loc='left')
        fig.colorbar(pol,ax=list(axes[:,1]),label=r'$s_3$',ticks=[-1,0,1],fraction=.045,pad=.02)
        fig.suptitle(r'TE–TM coupling, $t=3000$ ps',fontsize=10)
        return save(fig,out,'night_T0_polarization_review')


def make_precision_review_figure():
    """Numerical precision diagnostics from saved histories/frames; no simulation."""
    directory=DATA/'report3';names=('production_J1_ring','production_A1_spot')
    with plt.rc_context({**STYLE,'axes.titlesize':9,'font.size':8,'legend.fontsize':7}):
        fig,axes=plt.subplots(2,2,figsize=(WIDTH,110/25.4),layout='constrained')
        saved=[]
        for col,name in enumerate(names):
            with np.load(directory/(name+'.npz')) as z:hr=z['history'].copy();fr=z['features'].copy()
            with np.load(directory/(name+'_c64.npz')) as z:hs=z['history'].copy();fs=z['features'].copy()
            saved.append((hr,fr,hs,fs))
            for j,color in ((1,'#2166ac'),(2,'#b35806')):
                label=r'$N_'+('+' if j==1 else '-')+'$'
                axes[0,col].plot(hr[:,0],hr[:,j],color=color,lw=1.1,label=label+', 128')
                axes[0,col].plot(hs[:,0],hs[:,j],color=color,lw=.9,ls='--',label=label+', 64')
            axes[0,col].set(xlabel='Time (ps)',ylabel='Condensate population')
        handles,labels=axes[0,0].get_legend_handles_labels()
        fig.legend(handles,labels,loc='outside upper center',ncol=4,frameon=False)
        with np.load(directory/(names[0]+'_animation.npz')) as z:tr=z['time'];nr=z['density'].astype(float)
        with np.load(directory/(names[0]+'_c64_animation.npz')) as z:ts=z['time'];ns=z['density'].astype(float)
        tt,ir,isingle=np.intersect1d(tr,ts,return_indices=True)
        norm=np.linalg.norm(nr[ir].reshape(len(ir),-1),axis=1)
        error=np.linalg.norm((ns[isingle]-nr[ir]).reshape(len(ir),-1),axis=1)/np.maximum(norm,1e-30)
        keep=tt>0
        axes[1,0].semilogy(tt[keep],np.maximum(error[keep],1e-12),color='#762a83',lw=1.1)
        axes[1,0].axhline(.001,color='.5',ls=':',lw=.8)
        axes[1,0].set(xlabel='Time (ps)',ylabel=r'Relative density difference')
        _,fr,_,fs=saved[1];cr=fr[:,7]+1j*fr[:,8];cs=fs[:,7]+1j*fs[:,8]
        keep=(fr[:,0]>=400)&(abs(cr)>1.)
        axes[1,1].plot(fr[keep,0],np.angle(cs[keep]*cr[keep].conj()),color='#762a83',lw=1.1)
        axes[1,1].set(xlabel='Time (ps)',ylabel=r'Polarization phase difference (rad)')
        for ax,title in zip(axes.ravel(),('(a) Ring: populations','(b) Spot: populations',
                                          '(c) Ring: density maps','(d) Spot: polarization phase')):
            ax.set_title(title,loc='left');ax.spines[['top','right']].set_visible(False)
        return save(fig,ROOT/'results/diagnostics/report3','precision_comparison')


# Report 3: the saved production series, separate from the earlier N figures.
def report3_polar_profile(psi,x,limits=(5.,7.5),order=3):
    """One common polar quadrature; avoids unequal Cartesian pixel counts per sector."""
    from numpy.polynomial.legendre import leggauss
    from scipy.ndimage import map_coordinates
    radial,rw=leggauss(32);angular,aw=leggauss(8);width=2*np.pi/72
    r=(limits[0]+limits[1])/2+radial*(limits[1]-limits[0])/2
    centers=-np.pi+(np.arange(72)+.5)*width;phi=centers[:,None]+angular[None,:]*width/2
    xx=r[None,:,None]*np.cos(phi[:,None,:]);yy=r[None,:,None]*np.sin(phi[:,None,:]);dx=x[1]-x[0]
    coordinates=np.array([(yy-x[0])/dx,(xx-x[0])/dx])
    z=np.array([map_coordinates(v,coordinates,order=order,mode='grid-wrap') for v in psi])
    n=abs(z)**2;weights=r[None,:,None]*rw[None,:,None]*aw[None,None,:]
    total=np.sum(n.sum(0)*weights,axis=(1,2));difference=np.sum((n[0]-n[1])*weights,axis=(1,2))
    return np.degrees(centers),difference/total


def _report3_saved_gifs(out,load,manifest,*,only=None):
    """Density GIFs from portable archives; Stokes GIFs from saved complex probes."""
    from PIL import Image
    from model import stokes
    from numerics import write_json
    directory=DATA/'report3';records={}
    palette=Image.new('P',(1,1));palette.putpalette(np.round(255*np.vstack((plt.get_cmap('viridis')(np.linspace(0,1,112))[:,:3],plt.get_cmap('RdBu_r')(np.linspace(0,1,112))[:,:3],np.repeat(np.linspace(0,1,32)[:,None],3,axis=1)))).astype('uint8').ravel().tolist())
    def render(key,times,frames,x,titles,ranges,caption,source):
        times=np.asarray(times);durations=np.diff(np.rint(np.linspace(0,3000,len(times)+1)).astype(int))*10
        rendered=[];dx=x[1]-x[0];extent=[x[0]-dx/2,x[-1]+dx/2]*2
        with plt.rc_context(STYLE):
            fig,axes=plt.subplots(1,3,figsize=(9,3.5),dpi=95,layout='constrained');artists=[]
            for j,ax in enumerate(axes):
                cmap=plt.get_cmap('viridis' if ranges[j][0]==0 else 'RdBu_r').copy();cmap.set_bad('#ececec')
                artists.append(ax.imshow(np.zeros((len(x),len(x))),origin='lower',extent=extent,cmap=cmap,vmin=ranges[j][0],vmax=ranges[j][1],interpolation='nearest'))
                ax.set(title=titles[j],xlabel=r'$x$ ($\mu$m)',ylabel=r'$y$ ($\mu$m)',xticks=[-20,0,20],yticks=[-20,0,20]);ax.grid(False)
                fig.colorbar(artists[j],ax=ax,fraction=.046,pad=.025)
            geometry='Gaussian spot' if 'spot' in source else 'Annular pump'
            case='C1' if 'C1' in source else 'T0'
            label=case+' | '+geometry+(' | half-step control' if 'dt0125' in source else '')
            heading=fig.suptitle(label+' | t=0 ps',fontsize=10)
            fig.canvas.draw();fig.set_layout_engine('none')
            for i,(t,values) in enumerate(zip(times,frames)):
                for im,v in zip(artists,values):im.set_data(v)
                heading.set_text(label+f' | t={t:g} ps')
                if i in (0,len(times)-1):_layout_check(fig,key)
                fig.canvas.draw();frame=Image.fromarray(np.asarray(fig.canvas.buffer_rgba())[:,:,:3]);rendered.append(frame.quantize(palette=palette,dither=Image.Dither.NONE));frame.close()
            path=out/(key+'.gif');tmp=path.with_suffix('.tmp.gif')
            rendered[0].save(tmp,save_all=True,append_images=rendered[1:],duration=durations.tolist(),loop=0,optimize=False,disposal=2);tmp.replace(path)
            plt.close(fig)
        for frame in rendered:frame.close()
        records[key]={'file':str(path.relative_to(ROOT)),'source':source,'caption_en':caption,'times_ps':times.tolist(),'duration_ms':int(durations.sum()),'frames':len(times),'interpolation':'none; saved frames only','ranges':ranges,'color_maps':['viridis' if v[0]==0 else 'RdBu_r' for v in ranges]}
        manifest['animations'].update(records);write_json(directory/'figure_manifest.json',manifest)
        print('GIF',path.name,len(times),'frames',flush=True)
    density_cases=[('T0','spot','production_T0_spot'),('T0','ring','matched_T0_ring'),('C1','spot','production_C1_spot')]
    for case,geom,name in density_cases:
        key=case+'_'+geom+'_density_evolution'
        if only is not None and key not in only:continue
        with np.load(directory/(name+'_animation.npz'),allow_pickle=False) as z:t=z['time'];n=z['density'];x=z['x']
        # Actual saved sampling is retained; every frame has equal physical spacing.
        ix=np.flatnonzero(abs(x)<=25);n=n[:,:,ix[:,None],ix];x=x[ix];peak=float(n.max());total=n.sum(1)
        s3=np.ma.masked_where(total<.001*total.max(axis=(1,2))[:,None,None],(n[:,0]-n[:,1])/np.maximum(total,1e-30))
        frames=((n[i,0],n[i,1],s3[i]) for i in range(len(t)))
        render(key,t,frames,x,[r'$n_+$',r'$n_-$',r'Circular $s_3$'],[(0,peak),(0,peak),(-1,1)],'Component-density evolution from saved20-ps frames. One fixed density scale for both components and all times; circular polarization uses [-1,1]. Ring evolution is a finite-time record, not proof of stationary full polarization.',name)
    for case,geom,name in [('C1','spot','production_C1_spot'),('C1','ring','matched_C1_ring'),('T0','ring','matched_T0_ring')]:
        key=case+'_'+geom+'_Stokes_evolution'
        if only is not None and key not in only:continue
        compact_source=directory/'stokes_animation_compact.npz'
        if compact_source.exists():
            with np.load(compact_source,allow_pickle=False) as z:name=json.loads(str(z['metadata_json']))[case+'_'+geom]['source']
        # A compact archive records its source independently of cleared probe indexes.
        if case=='C1' and geom=='spot' and not compact_source.exists():
            candidate=ROOT/'results/runs/report3/check_C1_spot_full_dt0125'
            if (candidate/'summary.json').exists():
                summary=json.loads((candidate/'summary.json').read_text())
                if summary.get('run_status')=='RUN_COMPLETED' and summary.get('time_ps',0)>=3000:
                    with np.load(candidate/'checkpoint.npz') as z:ts=z['probe_times'];has_zero=np.any(np.isclose(ts,0,rtol=0,atol=1e-8))
                    if has_zero:name='check_C1_spot_full_dt0125'
        compact=directory/'stokes_animation_compact.npz'
        if compact.exists():
            prefix=case+'_'+geom
            with np.load(compact,allow_pickle=False) as z:
                info=json.loads(str(z['metadata_json']))[prefix]
                if info['source']!=name:raise ValueError('Compact Stokes source differs: '+name)
                t=z[prefix+'_time'].copy();x=z[prefix+'_x'].copy()
                values=z[prefix+'_stokes'].copy();masks=z[prefix+'_mask'].copy()
            frames=(tuple(np.ma.masked_where(masks[i],values[i,j]) for j in range(3)) for i in range(len(t)))
            render(key,t,frames,x,[r'Linear $s_1$',r'Diagonal $s_2$',r'Circular $s_3$'],[(-1,1)]*3,
                   'Normalized Stokes from actual saved complex fields. Full evolution from0 ps; low-intensity pixels are masked. Fixed scales [-1,1], no interpolation. The compact float64 source retains the original frame times and masks.',name)
            continue
        # Read-only memory map: plotting never opens the solver probe file in r+.
        run=ROOT/'results/runs/report3'/name
        with np.load(run/'checkpoint.npz',allow_pickle=False) as z:times=z['probe_times'].copy()
        maps=np.load(run/'probe_maps.npy',mmap_mode='r');valid=np.flatnonzero(np.isfinite(times));valid=valid[np.argsort(times[valid])]
        from_zero=bool(np.isclose(times[valid[0]],0.,rtol=0,atol=1e-8))
        if from_zero:
            stride=4 if geom=='ring' else 2
            valid=valid[np.unique(np.r_[np.arange(0,len(valid),stride),len(valid)-1])]
        else:
            last=times[valid[-1]];valid=valid[times[valid]>=last-170.-1e-8]
        d=load(name);x=d['x'];keep=np.flatnonzero(abs(x)<=25)
        def frames():
            for i in valid:
                v=maps[i];psi=np.stack((v[0]+1j*v[1],v[2]+1j*v[3]));S=stokes(psi);mask=S[0]<.001*S[0].max()
                yield tuple(np.ma.masked_where(mask,S[j]/np.maximum(S[0],1e-300))[np.ix_(keep,keep)] for j in (1,2,3))
        render(key,times[valid],frames(),x[keep],[r'Linear $s_1$',r'Diagonal $s_2$',r'Circular $s_3$'],[(-1,1)]*3,'Normalized Stokes from actual saved complex fields and the fixed circular-basis convention. Low-intensity pixels are masked, all color scales remain [-1,1]. No phase or frame interpolation. '+('Full evolution from0 ps. A time-step control, when used, is identified by its source run.' if from_zero else 'Only late complex fields survived for this old reference run; full early history awaits the independent time-step control.'),name)
    palette.close()
    return records


def wait_for_full_C1_spot_gif():
    """Low-priority CPU postprocessing after the already planned time-step control."""
    import time
    from numerics import write_json
    directory=DATA/'report3';queue_file=directory/'experiment_queue.json'
    while True:
        queue=json.loads(queue_file.read_text())
        record=queue.get('experiments',{}).get('C1_spot_dt',{})
        if record.get('status')=='COMPLETED':
            source=ROOT/'results/runs/report3/check_C1_spot_full_dt0125/checkpoint.npz'
            with np.load(source,allow_pickle=False) as z:
                if not np.any(np.isclose(z['probe_times'],0.,rtol=0,atol=1e-8)):
                    raise ValueError('Full complex history does not include t=0')
            manifest=json.loads((directory/'figure_manifest.json').read_text())
            def load(name):
                with np.load(directory/(name+'.npz'),allow_pickle=False) as z:return {'x':z['x'].copy()}
            _report3_saved_gifs(ROOT/'results/figures/report3',load,manifest,only={'C1_spot_Stokes_evolution'})
            manifest=json.loads((directory/'figure_manifest.json').read_text())
            manifest['full_C1_spot_gif_status']='COMPLETED_FROM_ZERO'
            write_json(directory/'figure_manifest.json',manifest)
            return
        if queue.get('status') in ('PAUSED','FAILED'):
            print('Full C1 spot GIF deferred: queue '+queue['status'],flush=True)
            return
        time.sleep(30.)


def make_spot_J_diagnostic():
    """TE-TM-only 2D J diagnostic; saved data, no integration."""
    import yaml
    directory=DATA/'report3';queue=json.loads((directory/'experiment_queue.json').read_text())
    records=[r for key,r in queue['experiments'].items() if key.startswith('Jspot_') and r.get('status')=='COMPLETED']
    records += [queue['experiments'][key] for key in ('C1_spot_dt','C1_spot_grid')]
    # A changed output cadence is not a new spatial/time integration accuracy.
    # Keep the densest history for duplicate (J,beta,N,L,dt) preparations.
    chosen={};excluded=[]
    for record in records:
        name=record['name'];c=yaml.safe_load((ROOT/'results/runs/report3'/name/'resolved_config.yaml').read_text())
        if c['physical']['beta']==0:continue
        key=tuple(c['physical'][k] for k in ('J','beta'))+(c['space']['N'],c['space']['L'],c['time']['dt'])
        if key in chosen and chosen[key][1]['save']['probe']<=c['save']['probe']:
            excluded.append(name);continue
        if key in chosen:excluded.append(chosen[key][0]['name'])
        chosen[key]=(record,c)
    rows=[]
    phase_archive=directory/'local_phase_histories.npz'
    phase_data=np.load(phase_archive,allow_pickle=False) if phase_archive.exists() else None
    for record,c in chosen.values():
        name=record['name'];review=record['review'];windows=review['windows'];run=ROOT/'results/runs/report3'/name
        with np.load(directory/(name+'.npz'),allow_pickle=False) as z:f=z['features'].copy()
        intervals=[w['interval_ps'] for w in windows];periods=review.get('periodic',{}).get('periods_ps')
        phase=-np.unwrap(np.angle(f[:,7]+1j*f[:,8]));n=f[:,1:3];imbalance=(n[:,0]-n[:,1])/n.sum(1)
        stats=[]
        for j,(left,right) in enumerate(intervals):
            if periods:
                cycles=int(np.floor((right-left)/periods[j]+1e-8));right=left+cycles*periods[j]
            else:cycles=None
            t=np.r_[left,f[(f[:,0]>left)&(f[:,0]<right),0],right]
            zz=np.interp(t,f[:,0],imbalance);pp=np.interp(t,f[:,0],phase)
            stats.append({'interval_ps':[left,right],'mean_Z':float(np.trapezoid(zz,t)/(right-left)),
                          'A_Z':float(np.ptp(zz)/2),'phase_drift_rad_ps':float((pp[-1]-pp[0])/(right-left)),
                          'phase_range_rad':float(np.ptp(pp)),
                          'winding_per_cycle':float((pp[-1]-pp[0])/(2*np.pi*cycles)) if cycles else None})
        # Independently inspect local relative phase where both components stay visible.
        maps=None
        if (run/'probe_maps.npy').exists():
            with np.load(run/'checkpoint.npz',allow_pickle=False) as z:times=z['probe_times'].copy()
            maps=np.load(run/'probe_maps.npy',mmap_mode='r',allow_pickle=False)
        else:
            if phase_data is None or name+'_time' not in phase_data:raise ValueError('Missing compact phase history: '+name)
            times=phase_data[name+'_time'];points=phase_data[name+'_psi']
        local=[];N=c['space']['N'];dx=c['space']['L']/N
        for point_index,(x,y) in enumerate(((0,0),(2,0),(-2,0),(0,2),(0,-2))):
            point=[]
            for left,right in intervals:
                ids=np.flatnonzero((times>=left-1e-8)&(times<=right+1e-8));ids=ids[np.argsort(times[ids])]
                if maps is not None:
                    v=np.asarray(maps[ids,:,round(N/2+y/dx),round(N/2+x/dx)]);plus=v[:,0]+1j*v[:,1];minus=v[:,2]+1j*v[:,3]
                else:plus,minus=points[ids,point_index].T
                arg=np.angle(plus*minus.conj());pp=np.unwrap(arg)
                fraction=np.minimum(abs(plus)**2,abs(minus)**2)/(abs(plus)**2+abs(minus)**2)
                point.append({'phase_range_rad':float(np.ptp(pp)),'turns':float((pp[-1]-pp[0])/(2*np.pi)),
                              'min_component_fraction':float(fraction.min()),'max_phase_increment_rad':float(max(abs(np.diff(pp))))})
            local.append({'xy_um':[x,y],'windows':point})
        masked=any(w['min_component_fraction']<1e-3 or w['max_phase_increment_rad']>.8*np.pi for p in local for w in p['windows'])
        rotating=all(abs(w['winding_per_cycle'] or 0)>.5 for w in stats) and all(abs(w['turns'])>1 for p in local for w in p['windows'])
        bounded=all(w['phase_range_rad']<2*np.pi and abs(w['winding_per_cycle'] or 0)<.05 for w in stats) and all(w['phase_range_rad']<2*np.pi for p in local for w in p['windows'])
        if not masked and review['status']=='STATIONARY' and max(w['phase_range_rad'] for p in local for w in p['windows'])<.01:regime='LOCKED'
        elif not masked and review['status']=='PERIODIC' and rotating:regime='RUNNING'
        elif not masked and review['status']=='PERIODIC' and bounded and max(w['A_Z'] for w in stats)>.001:regime='BOUNDED'
        else:regime='UNRESOLVED'
        rows.append({'name':name,'J_meV':c['physical']['J'],'beta':c['physical']['beta'],'N':N,'L_um':c['space']['L'],
                     'dt_ps':c['time']['dt'],'precision':c.get('execution',{}).get('precision','complex128'),
                     'probe_ps':c['save']['probe'],'regime':regime,'original_status':review['status'],
                     'statistics':stats,'local_phase_checks':local,'numerical_convergence':record.get('convergence'),
                     'mean_Z':float(np.mean([w['mean_Z'] for w in stats])),
                     'A_Z':float(np.mean([w['A_Z'] for w in stats])),
                     'phase_drift_rad_ps':float(np.mean([w['phase_drift_rad_ps'] for w in stats]))})
    styles={'RUNNING':('C1','Running phase'),'BOUNDED':('C2','Oscillating phase'),'LOCKED':('C0','Fixed phase'),'UNRESOLVED':('.5','Unresolved')}
    def resolution(r):return (r['N'],r['L_um'],r['dt_ps'])
    symbols={(256,256.,.025):('o',r'$256^2$, $L=256$, $\Delta t=.025$'),
             (512,256.,.025):('s',r'$512^2$, $L=256$, $\Delta t=.025$'),
             (512,256.,.0125):('v',r'$512^2$, $L=256$, $\Delta t=.0125$'),
             (768,256.,.025):('^',r'$768^2$, $L=256$, $\Delta t=.025$'),
             (768,384.,.025):('h',r'$768^2$, $L=384$, $\Delta t=.025$'),
             (384,384.,.025):('D',r'$384^2$, $L=384$, $\Delta t=.025$'),
             (512,512.,.025):('P',r'$512^2$, $L=512$, $\Delta t=.025$')}
    out=ROOT/'results/diagnostics/report3'
    with plt.rc_context(STYLE):
        fig,axes=plt.subplots(3,1,figsize=(WIDTH,150/25.4),layout='constrained',sharex=True,squeeze=False)
        for col,beta in enumerate((.0119,)):
            for r in sorted(rows,key=lambda r:r['N']):
                if r['beta']!=beta:continue
                color,_=styles[r['regime']];marker,_=symbols[resolution(r)]
                for row,key in enumerate(('mean_Z','A_Z','phase_drift_rad_ps')):
                    axes[row,col].plot(1000*r['J_meV'],r[key],marker=marker,ls='',ms=5.5,
                                       mec=color,mfc='none',mew=.85,alpha=.85)
            axes[0,col].set_title(r'TE-TM coupling: $\beta=0.0119$ meV $\mu$m$^2$',loc='left')
            axes[2,col].set_xlabel(r'Coherent coupling, $J$ ($\mu$eV)')
            for ax in axes[:,col]:ax.set_xlim(-3,123);ax.set_xticks([0,20,40,60,80,100,120]);ax.axhline(0,color='.55',lw=.65,zorder=0)
        axes[0,0].set_ylabel(r'Mean $Z$')
        axes[1,0].set_ylabel(r'Amplitude $A_Z$')
        axes[2,0].set_ylabel(r'Phase drift (rad ps$^{-1}$)')
        for ax in axes[:,0]:ax.yaxis.set_label_coords(-.14,.5)
        for ax in axes[1]:ax.set_ylim(-.025,.77)
        regimes=[Line2D([],[],color=color,lw=2,label=label) for key,(color,label) in styles.items() if any(r['regime']==key for r in rows)]
        fig.legend(handles=regimes,loc='outside upper center',ncol=2,frameon=False,title='Diagnostic 2D scan: Gaussian spot')
        handles=[Line2D([],[],color='.25',mfc='none',ls='',marker=m,label=label,ms=5) for key,(m,label) in symbols.items() if any(resolution(r)==key for r in rows)]
        fig.legend(handles=handles,loc='outside lower center',ncol=2,fontsize=7,frameon=False,title=r'Numerical controls: $L$ in $\mu$m, $\Delta t$ in ps; complex128')
        fig.set_layout_engine('none')
        for k,ax in enumerate(axes[:,0]):ax.set_position([.18,.27+(2-k)*.19,.80,.16])
        path=save(fig,out,'spot_J_regime_scan')
    report={'status':'DIAGNOSTIC_ONLY','figure':str(path.relative_to(ROOT)),
            'definitions':{'Z':'(Nplus-Nminus)/(Nplus+Nminus)','A_Z':'half peak-to-peak Z; averaged across two windows',
                           'theta':'-unwrap(arg(integral S1+iS2)); local checks use arg(psi_plus*conj(psi_minus))',
                           'means':'integer-cycle trapezoidal means in each accepted window; stationary cases use full windows',
                           'scope':'TE-TM series only. Three sampled finite-time regimes; five bright-point phase checks, not uniform phase or full stability proof. No interpolated transition boundaries. Symbols show spatial/time accuracy; all arithmetic complex128. Domain L256 remains imperfect for polarization near J=.02.'},
            'duplicate_preparations_excluded':excluded,'cases':rows}
    from numerics import write_json
    write_json(directory/'spot_J_diagnostic.json',report)
    return path


def make_report3_figures(*,animations=False):
    """The 16-figure report set from local/figure_requirements.md; no report editing."""
    import hashlib
    import shutil
    from model import stokes
    from numerics import sample_report3_center_cycles,write_json
    from matplotlib.colors import Normalize
    from matplotlib.collections import LineCollection
    from mpl_toolkits.mplot3d.art3d import Line3DCollection
    directory=DATA/'report3';out=ROOT/'results/figures/report3';out.mkdir(parents=True,exist_ok=True)
    manifest_path=directory/'figure_manifest.json';old=json.loads(manifest_path.read_text())
    manifest={'method':'Saved arrays and verified short CPU replay of two local cycles; report text unchanged',
              'PNG_width_mm':160,'PNG_dpi':300,'mask_relative_S0':.001,'figures':{},'animations':old.get('animations',{}),
              'figure_order':['P1_reservoir_switching','P2_homogeneous_pump_response','N1_spatial_component_densities',
                 'P3_homogeneous_densities','P4_homogeneous_relative_phase','P5_homogeneous_J_scan',
                 'N3_TE_TM_spot','N4_TE_TM_ring','N5_TE_TM_angular_response',
                 'N7_ring_angular_polarization','N12_spot_J_scan','N13_spot_regime_dynamics','N8_spot_Stokes','N9_ring_Stokes','N10_homogeneous_Poincare','N11_spot_Poincare'],
              'limitations':['T0 ring at6000ps is a finite-time polarization snapshot','Alpha2 figure is reserved; no paired error bound',
                             'The diagnostic2D J scan is separate from this report set']}
    cache={};paths=[]
    with np.load(sample_report3_center_cycles(),allow_pickle=False) as z:
        cycles={k:z[k].copy() for k in z.files if k!='metadata_json'};local_meta=json.loads(str(z['metadata_json']))
    reviews=json.loads((directory/'report3_cycle_review.json').read_text())['cases']
    from numerics import _lag_polynomial_weights
    center_cases=[]
    queue=json.loads((directory/'experiment_queue.json').read_text())
    center_archive=directory/'regime_center_data.npz'
    if center_archive.exists():
        with np.load(center_archive,allow_pickle=False) as z:
            center_cases=json.loads(str(z['metadata_json']))
            for i,c in enumerate(center_cases):
                for k in ('time','psi','stokes','phase','z','initial_psi'):c[k]=z[f'{i}_{k}'].copy()
    else:
        for label,name in [('Running phase','Jspot_TETM_J0p04000_refine'),('Oscillating phase','Jspot_TETM_J0p06500_fill2h'),('Fixed phase','Jspot_TETM_J0p10000_fill2h')]:
            from numerics import _lag_polynomial_weights
            run=ROOT/'results/runs/report3'/name
            with np.load(run/'checkpoint.npz',allow_pickle=False) as z:ts=z['probe_times'].copy()
            raw=np.load(run/'probe_maps.npy',mmap_mode='r');ids=np.flatnonzero(np.isfinite(ts));ids=ids[np.argsort(ts[ids])];t=ts[ids];n=raw.shape[-1]//2
            v=np.asarray(raw[ids,:4,n,n]);psi=np.stack((v[:,0]+1j*v[:,1],v[:,2]+1j*v[:,3]),axis=1)
            S=stokes(psi,axis=1);vec=(S[1:]/S[0]).T;phase=np.unwrap(np.angle(psi[:,0]*psi[:,1].conj()));review=queue['experiments'][name]['review'];T=review.get('period_ps');errors=[];cross=[]
            if T:
                for j in np.flatnonzero((t>=2500)&(t<=2700))[::8]:
                    vals=[]
                    for count in (6,8):
                        ix,w=_lag_polynomial_weights(t,t[j]+T,count);vals.append(w@vec[ix])
                    errors.append(float(np.linalg.norm(vals[1]-vec[j])));cross.append(float(np.linalg.norm(vals[0]-vals[1])))
                if max(errors)>.01 or max(cross)>.002:raise ValueError('Local cycle not confirmed: '+name)
            start=np.flatnonzero(np.isclose(t,2600.,atol=1e-8,rtol=0));assert len(start)==1
            field=raw[ids[start[0]]];initial=np.stack((field[0]+1j*field[1],field[2]+1j*field[3]))
            center_cases.append(dict(name=name,label=label,time=t,psi=psi,stokes=vec,phase=phase,z=S[3]/S[0],period=T,initial_psi=initial,review_status=review['status'],recurrence_max=max(errors,default=0.),interpolation_crosscheck_max=max(cross,default=0.)))
        values={f'{i}_{k}':c[k] for i,c in enumerate(center_cases) for k in ('time','psi','stokes','phase','z','initial_psi')}
        meta=[{k:v for k,v in c.items() if k not in ('time','psi','stokes','phase','z','initial_psi')} for c in center_cases]
        values['metadata_json']=np.array(json.dumps(meta));np.savez_compressed(center_archive,**values)
    for c in center_cases:
        T=c['period']
        if not T:continue
        c['full_map_period_ps']=T
        for _ in range(3):
            candidate=T/2;errors=[];cross=[]
            if candidate<8*np.median(np.diff(c['time'])):break
            for j in np.flatnonzero((c['time']>=2500)&(c['time']<=2700))[::8]:
                values=[]
                for count in (6,8):
                    ix,w=_lag_polynomial_weights(c['time'],c['time'][j]+candidate,count);values.append(w@c['stokes'][ix])
                errors.append(float(np.linalg.norm(values[1]-c['stokes'][j])));cross.append(float(np.linalg.norm(values[0]-values[1])))
            if max(errors)>.01 or max(cross)>.002:break
            T=candidate;c['period']=T;c['recurrence_max']=max(errors);c['interpolation_crosscheck_max']=max(cross)
    def load(name):
        if name not in cache:
            with np.load(directory/(name+'.npz'),allow_pickle=False) as z:
                d={k:z[k].copy() for k in ('psi','x','y','features','history','mean_maps','mean_interval_ps')}
                d.update(config=json.loads(str(z['config_json'])),summary=json.loads(str(z['summary_json'])))
            d.update(name=name,time_ps=float(d['history'][-1,0]));cache[name]=d
        return cache[name]
    def config_caption(d):
        c=d['config'];p=parameters(c)
        return (f"{d['name']}: p={c['pump']['P0']:g}, epsilon={c['pump']['epsilon']:g}, J={p['J']:g} meV, "
                f"beta={p['beta']:g} meV um^2, alpha2={p['alpha2']:g} meV um^2, "
                f"N={c['space']['N']}, L={c['space']['L']:g} um, dt={c['time']['dt']:g} ps. ")
    def publish(path,key,caption,members=(),status='READY',**extra):
        manifest['figures'][key]={'file':str(path.relative_to(ROOT)),'caption_en':caption,'status':'DIAGNOSTIC_ONLY' if key=='N6_spot_coupling_dynamics' else status,
            'sources':[{'run':d['name'],'time_ps':d['time_ps'],'config':d['config'],
                        'field_sha256':hashlib.sha256(np.ascontiguousarray(d['psi']).tobytes()).hexdigest()} for d in members],**extra}
        paths.append(path);print('REPORT FIGURE',key,status,flush=True)
    def finish(fig,key,caption,members=(),height=72,**extra):
        fig.set_size_inches(WIDTH,height/25.4);width=105 if key=='N5_TE_TM_angular_response' else 160
        target=ROOT/'results/diagnostics/report3' if key=='N6_spot_coupling_dynamics' else out
        path=save(fig,target,key,width_mm=width);publish(path,key,caption,members,**extra)
    def draw(ax,image,d,title,polar=False,vmax=None,center=False):
        x=d['x'];ids=np.flatnonzero(abs(x)<=25);dx=x[1]-x[0];extent=[x[ids[0]]-dx/2,x[ids[-1]]+dx/2]*2
        image=image[np.ix_(ids,ids)];S0=abs(d['psi'])**2;S0=S0.sum(0)
        if polar:image=np.ma.masked_where(S0[np.ix_(ids,ids)]<.001*S0.max(),image)
        cmap=plt.get_cmap('RdBu_r' if polar else 'viridis').copy();cmap.set_bad('#ececec')
        im=ax.imshow(image,origin='lower',extent=extent,cmap=cmap,vmin=-1 if polar else 0,vmax=1 if polar else vmax,interpolation='nearest')
        profile=pump_profile(d['config'],make_grid(d['config']))
        ax.contour(x[ids],x[ids],profile[np.ix_(ids,ids)],levels=[.5],colors='.2' if polar else 'white',linewidths=.55)
        if center:ax.plot(0,0,'o',ms=2.5,mfc='black',mec='white',mew=.55)
        ax.set(title=title,xlabel=r'$x$ ($\mu$m)',ylabel=r'$y$ ($\mu$m)',aspect='equal',xticks=[-20,0,20],yticks=[-20,0,20]);ax.grid(False)
        return im
    def draw_projection(ax,v,fixed=False,fractions=None):
        ax.set_xlim(-1.08,1.08);ax.set_ylim(-1.08,1.08);ax.set_aspect('equal');ax.grid(False)
        circle=np.linspace(0,2*np.pi,200)
        ax.plot(np.cos(circle),np.sin(circle),color='.86',lw=.45)
        ax.axhline(0,color='.78',lw=.35);ax.axvline(0,color='.78',lw=.35)
        if fixed:
            ax.scatter(v[-1,0],v[-1,1],color='C0',s=9,zorder=5)
        else:
            points=v[:,:2];segments=np.stack((points[:-1],points[1:]),axis=1)
            fraction=np.linspace(0,1,len(v)) if fractions is None else np.asarray(fractions)
            collection=LineCollection(segments,cmap='viridis',norm=Normalize(0,1),lw=1.0)
            collection.set_array((fraction[:-1]+fraction[1:])/2);ax.add_collection(collection)
            ax.scatter(points[0,0],points[0,1],color='C1',s=8,zorder=6)
        ax.set_xticks([-1,0,1]);ax.set_yticks([-1,0,1]);ax.tick_params(labelsize=5,pad=1,length=1.5,width=.5)
        ax.set_xlabel(r'$s_1$',fontsize=6,labelpad=0);ax.set_ylabel(r'$s_2$',fontsize=6,labelpad=0)

    def sphere(ax,v,title,fixed=False,fractions=None):
        from mpl_toolkits.mplot3d import proj3d
        ax.set(xlim=(-1.2,1.2),ylim=(-1.2,1.2),zlim=(-1.2,1.2));ax.set_box_aspect((1,1,1),zoom=1.12);ax.view_init(elev=22,azim=35)
        u=np.linspace(0,2*np.pi,25);w=np.linspace(0,np.pi,13)
        ax.plot_wireframe(np.outer(np.cos(u),np.sin(w)),np.outer(np.sin(u),np.sin(w)),np.outer(np.ones_like(u),np.cos(w)),color='.83',lw=.3,alpha=.3)
        for k,label in enumerate((r'$s_1$',r'$s_2$',r'$s_3$')):
            line=np.zeros((2,3));line[1,k]=1.12;ax.plot(*line.T,color='.5',lw=.5);tip=line[1].copy()*1.07;px,py,_=proj3d.proj_transform(*tip,ax.get_proj());ax.text2D(px,py,label,transform=ax.transData,fontsize=8)
        if fixed:ax.scatter(*v[-1],color='C0',s=24,depthshade=False)
        else:
            segments=np.stack((v[:-1],v[1:]),axis=1);collection=Line3DCollection(segments,cmap='viridis',norm=Normalize(0,1),lw=1.7)
            fraction=np.linspace(0,1,len(v)) if fractions is None else np.asarray(fractions)
            collection.set_array((fraction[:-1]+fraction[1:])/2);ax.add_collection3d(collection)
            ax.scatter(*v[0],color='C1',s=15,depthshade=False,zorder=5)
            k=max(1,len(v)//3);delta=v[min(len(v)-1,k+max(2,len(v)//40))]-v[k-1];length=np.linalg.norm(delta)
            if length>1e-10:ax.quiver(*v[k],*(delta/length*.13),color=plt.get_cmap('viridis')(k/(len(v)-1)),arrow_length_ratio=.6,lw=1.2)
        ax.set(xlim=(-1.2,1.2),ylim=(-1.2,1.2),zlim=(-1.2,1.2));ax.set_box_aspect((1,1,1),zoom=1.12);ax.view_init(elev=22,azim=35)
        ax.set_axis_off();ax.set_title(title,fontsize=9,pad=0)
        inset=ax.inset_axes([0.65,0.02,0.34,0.34])
        draw_projection(inset,v,fixed=fixed,fractions=fractions)
    with plt.rc_context(STYLE):
        reused=[('N1_reservoir_switching','P1_reservoir_switching','Matched scalar models after pump turn-on and turn-off, PA=1.5 Pth. Curves and dotted steady-state level are copied unchanged from the preceding report.'),
                ('N2_circular_populations','P2_homogeneous_pump_response','Homogeneous circular densities versus pump polarization epsilon, J=0 and total active pump4 Pth. Analytic stationary branches and independent integrations, copied unchanged from the preceding report.'),
                ('N4_component_dynamics','P3_homogeneous_densities','Homogeneous densities at J=5,6.5,10 microeV: running phase, bounded-phase oscillations and fixed phase. Same original trajectories, pump s=2, epsilon=.3 and common late interval as the preceding report.'),
                ('N5_relative_phase','P4_homogeneous_relative_phase','Relative phase theta=arg(psi+)-arg(psi-) for the same three homogeneous trajectories and late intervals as P3. Only integer2pi offsets are applied. Copied unchanged from the preceding report.'),
                ('N6_coupling_summary','P5_homogeneous_J_scan','Oscillation amplitude and mean frequency difference versus J for the original homogeneous preparation. Masked low-component-density phases and sampled transition interval retain the preceding report definitions; this scan is not combined with the new2D scan.')]
        fresh=[report_figure1(out),report_figure2(out),*report_figures45(out),report_figure6(out)]
        fresh_names=dict(zip((row[1] for row in reused),fresh))
        for source,key,caption in reused:
            if key=='P1_reservoir_switching':caption+=' The horizontal dotted line is the matched pumped stationary condensate density, before switch-off.'
            if key=='P2_homogeneous_pump_response':caption+=' Pth=gammaC*gammaA/R is the active-pump threshold per component. Spin exchange between reservoirs remains present even at J=0.'
            if key=='P5_homogeneous_J_scan':caption='Original homogeneous scan: amplitude and mean theta drift, theta=arg(psi+)-arg(psi-), so frequency difference is minus theta drift. Gray diamonds mark phase-masked cycles near weak component density; gaps do not mean zero.'
            path=out/(key+'.png');shutil.copy2(fresh_names[key],path)
            publish(path,key,caption,source_method={'P1_reservoir_switching':'report_figure1','P2_homogeneous_pump_response':'report_figure2','P3_homogeneous_densities':'report_figures45','P4_homogeneous_relative_phase':'report_figures45','P5_homogeneous_J_scan':'report_figure6'}[key],preserved_from_previous_report=key.startswith(('P1_','P2_')))
        for path in fresh:path.unlink()
        members=[load('production_A1_spot'),load('matched_A1_ring')]
        fig,axes=plt.subplots(2,2,figsize=(WIDTH,112/25.4),layout='constrained')
        caption='Circular-component densities without coherent coupling, J=beta=0, alpha2=-.1alpha1 and epsilon=.3. '
        for row,d in enumerate(members):
            n=abs(d['psi'])**2
            for k in (0,1):im=draw(axes[row,k],n[k],d,f'({chr(97+2*row+k)}) '+('Gaussian spot' if row==0 else 'Annular pump')+(': $n_+$' if k==0 else ': $n_-$'),vmax=float(n.max()))
            axes[row,1].set_ylabel('');axes[row,1].tick_params(labelleft=False)
            fig.colorbar(im,ax=axes[row],fraction=.035,pad=.02,label=r'Density ($\mu$m$^{-2}$)')
            pop=n.sum((-2,-1))*(d['x'][1]-d['x'][0])**2
            caption+=config_caption(d)+f"Snapshot t={d['time_ps']:g} ps, N+={pop[0]:.6g}, N-={pop[1]:.6g}, Z={(pop[0]-pop[1])/pop.sum():.6g}. "
        caption+='Densities share an absolute scale within each geometry. Half-maximum pump contours are shown. Stationary density does not imply stationary full polarization.'
        finish(fig,'N1_spatial_component_densities',caption,members,height=112)
        for key,name,geometry in [('N3_TE_TM_spot','production_T0_spot','Gaussian spot'),('N4_TE_TM_ring','matched_T0_ring','Annular pump')]:
            d=load(name);S=stokes(d['psi']);fig,axes=plt.subplots(1,2,figsize=(WIDTH,70/25.4),layout='constrained')
            im=draw(axes[0],S[0],d,'(a) Total density $S_0$',vmax=float(S[0].max()));fig.colorbar(im,ax=axes[0],fraction=.045,pad=.025,label=r'$S_0$ ($\mu$m$^{-2}$)')
            im=draw(axes[1],S[3]/np.maximum(S[0],1e-300),d,'(b) Circular polarization $s_3$',polar=True);fig.colorbar(im,ax=axes[1],fraction=.045,pad=.025,ticks=[-1,0,1])
            axes[1].set_ylabel('');axes[1].tick_params(labelleft=False)
            caption=geometry+f": snapshot at t={d['time_ps']:g} ps under equal component pumping epsilon=0. "+config_caption(d)
            caption+='The controlled beta=0 preparation is the corresponding B0 run; no change of epsilon is used to attribute polarization to TE-TM. Contours: half-maximum pump; s3 masked below0.001 peak S0. '
            caption+=('Ring polarization continues to evolve; this is not a phase-locked or stationary polarization state.' if geometry=='Annular pump' else 'Late density and polarization pass stationarity checks.')
            finish(fig,key,caption,[d],height=70,status='READY_FINITE_TIME' if geometry=='Annular pump' else 'READY')
        names=['production_B0_spot','report3_T0_spot_beta_0.5','production_T0_spot','report3_T0_spot_beta_2'];members=[load(n) for n in names]
        fig,ax=plt.subplots(figsize=(WIDTH,67/25.4),layout='constrained');means=[];ranges=[];windows={}
        for d in members:
            f=d['features'];f=f[f[:,0]>=f[-1,0]-400];q=2*np.hypot(f[:,10],f[:,11]);m=np.trapezoid(q,f[:,0])/(f[-1,0]-f[0,0]);means.append(m);ranges.append([m-q.min(),q.max()-m]);windows[d['name']]=[float(f[0,0]),float(f[-1,0])]
        ax.errorbar([0,.5,1,2],means,yerr=np.asarray(ranges).T,fmt='o-',ms=4,capsize=2,lw=1,color='C0')
        ax.set(xlabel=r'TE-TM coefficient, $\beta/\beta_{\rm ref}$',ylabel=r'Second harmonic, $Q_2$',xticks=[0,.5,1,2],ylim=(-.005,.34),xlim=(-.08,2.08))
        finish(fig,'N5_TE_TM_angular_response','Gaussian spot with equal pumping, J=0: Q2=2|C2|, C2=integral_R S3 exp(-2i phi) dA / integral_R S0 dA, R=5-15um. Four actual beta values; line is a guide only. Means over the final400ps; bars show temporal range, not numerical uncertainty. Same Cartesian quadrature for every point. The response is approximately linear over the sampled range; lines guide the eye. beta_ref=.0119meV um^2; p=13.8, epsilon=0, identical radial preparation.',members,height=52,windows_ps=windows,PNG_width_mm=105)
        coupling_labels={'A1':'No coherent coupling','T1':'TE-TM only','J1':'J only','C1':'J + TE-TM'}
        fig,ax=plt.subplots(figsize=(WIDTH,72/25.4),layout='constrained');members=[];begin=2600.;end=begin+3*max(reviews['production_'+k+'_spot']['period_ps'] for k in ('A1','T1','J1','C1'))
        for case,color,ls in [('A1','.35',':'),('T1','C0','-'),('J1','C2','--'),('C1','C1','-')]:
            d=load('production_'+case+'_spot');members.append(d);f=d['features'];keep=(f[:,0]>=begin)&(f[:,0]<=end)
            ax.plot(f[keep,0]-begin,2*f[keep,10],color=color,ls=ls,label=coupling_labels[case],lw=1.2)
        ax.set(xlabel=r'Time after $t_0=2600$ ps (ps)',ylabel=r'Cosine angular component, $2\mathrm{Re}\,C_2$',xlim=(0,end-begin))
        fig.legend(loc='outside upper center',ncol=2,frameon=False)
        finish(fig,'N6_spot_coupling_dynamics','Gaussian spot, epsilon=.3, p=13.8: 2Re C2 is the cosine component of the second angular harmonic of S3, normalized by S0 in R=5-15um. Approximately three late cycles are displayed from one shared t0=2600ps; no independent curve shifts. beta is0 or.0119meV um^2 and J is0 or.010meV. The beta=0 controls nearly overlap at zero. This observable alone does not prove rigid rotation of a whole polarization map.',members,height=72,display_interval_ps=[begin,end])
        fig,ax=plt.subplots(figsize=(WIDTH,72/25.4),layout='constrained');members=[];errors={}
        for case,color,ls in [('A1','.35',':'),('T1','C0','-'),('J1','C2','--'),('C1','C1','-')]:
            name='charge_J1_seeded_c128_L256_t3600' if case=='J1' else 'matched_'+case+'_ring';d=load(name);members.append(d)
            phi,v=report3_polar_profile(d['psi'],d['x']);_,cross=report3_polar_profile(d['psi'],d['x'],order=5);errors[name]=float(max(abs(v-cross)))
            if errors[name]>.001:raise ValueError('Polar interpolation requires refinement')
            ax.plot(phi,v,color=color,ls=ls,label=coupling_labels[case])
        ax.set(xlabel=r'Azimuthal angle, $\varphi$ (deg)',ylabel=r'Circular polarization, $s_3(\varphi)$',xlim=(-180,180),xticks=[-180,-90,0,90,180])
        fig.legend(loc='outside upper center',ncol=2,frameon=False)
        caption='Annular pump p=3.3, epsilon=.3: ratio of radial integrals S3/S0 in r=5-7.5um using a common72-sector polar Gauss quadrature, including r dr dphi. This is not an unweighted mean of local s3. '
        caption+=' '.join(f"{d['name']}: t={d['time_ps']:g}ps." for d in members)+' T1 is a snapshot of a periodic state, not a time-independent angular profile. Stationary references use their own stated late times; no common-time convergence is implied.'
        finish(fig,'N7_ring_angular_polarization',caption,members,height=72,polar_order3_vs5_errors=errors)
        for geom,name,key in [('spot',center_cases[0]['name'],'N8_spot_Stokes'),('ring','matched_C1_ring','N9_ring_Stokes')]:
            d=load(name).copy()
            if geom=='spot':d.update(psi=center_cases[0]['initial_psi'],time_ps=2600.)
            S=stokes(d['psi']);fig,axes=plt.subplots(1,3,figsize=(WIDTH,64/25.4),layout='constrained')
            for k,(ax,label) in enumerate(zip(axes,('Linear (H/V), $s_1$',r'Linear ($\pm45^\circ$), $s_2$','Circular, $s_3$')),1):
                im=draw(ax,S[k]/np.maximum(S[0],1e-300),d,f'({chr(96+k)}) '+label,polar=True,center=geom=='spot')
                if k>1:ax.set_ylabel('');ax.tick_params(labelleft=False)
            fig.colorbar(im,ax=axes,fraction=.025,pad=.02,ticks=[-1,0,1],label='Normalized Stokes')
            caption=f"Full local Stokes polarization of the {geom} at t={d['time_ps']:g}ps. "+config_caption(d)
            caption+='S1=2Re(conj(psi+)psi-), S2=2Im(conj(psi+)psi-), S3=n+-n-, si=Si/S0. Common scale[-1,1]; low-S0 pixels masked below0.001 peak. Contours show half-maximum pump. '
            caption+=('Instantaneous snapshot of the selected running-phase state; the center dot is the observation point in N11/N13 and this is their common initial time.' if geom=='spot' else 'Verified stationary density and polarization; any low-density central region is masked by the same rule.')
            finish(fig,key,caption,[d],height=64)
        scan=json.loads((DATA/'F5_scan.json').read_text());fig=plt.figure(figsize=(WIDTH,77/25.4),layout='constrained');axes=[];zero_sources=[]
        for i,(J,label) in enumerate([(.005,'Running phase'),(.0065,'Oscillating phase'),(.010,'Fixed phase')]):
            row=next(r for r in scan if abs(r['J']-J)<1e-10);data=load_0d(DATA/f'F5_J_{J:.5f}.npz');t=data['time'];T=row.get('period_ps');t0=17000.
            ids=np.flatnonzero((t>=t0)&(t<=t0+T)) if T else np.array([np.argmin(abs(t-18000.))])
            field=data['psi'][ids];S=stokes(field,axis=1);v=(S[1:]/S[0]).T;res=float(max(abs(np.sum(v*v,axis=1)-1)))
            if res>1e-12:raise ValueError('Invalid0D Stokes identity')
            ax=fig.add_subplot(1,3,i+1,projection='3d');axes.append(ax);sphere(ax,v,rf'$J={J*1000:g}$ $\mu$eV'+'\n'+label,fixed=T is None)
            zero_sources.append({'J_meV':J,'config':data['config'],'interval_ps':[float(t[ids[0]]),float(t[ids[-1]])],'period_ps':T,
                                 'samples':len(ids),'sphere_identity_max_residual':res,'min_component_fraction':float(np.min(abs(field)**2/np.sum(abs(field)**2,axis=1)[:,None])),'max_unwrapped_phase_increment':float(np.max(abs(np.diff(np.unwrap(np.angle(field[:,0]*field[:,1].conj())))),initial=0.)),'status':row['status']})
        bar=plt.cm.ScalarMappable(norm=Normalize(0,1),cmap='viridis');fig.colorbar(bar,ax=axes,orientation='horizontal',fraction=.045,pad=.02,label=r'$(t-t_0)/T$ (periodic trajectories only)',ticks=[0,.5,1])
        finish(fig,'N10_homogeneous_Poincare','Three original homogeneous J preparations on the Poincare sphere; s=2, epsilon=.3, beta=0. Running and bounded phase states show one late cycle at the actual0.5ps saved sampling; the fixed state is a point. s is computed from complex fields using S2=2Im(conj(psi+)psi-), not legacy_s2. No forced normalization or curve smoothing. Orange dot: start; arrow: direction; color: fraction of cycle. Measured periods:344.479051ps (running) and282.408941ps (oscillating); actual sampled intervals are recorded below. The oscillating0D component minimum fraction is0.00384 and the largest unwrapped increment is0.0723rad at0.5ps sampling, so the near-pole passage does not break the phase unwrap. A closed Stokes loop alone does not distinguish phase winding; retain the original theta(t) examples.',height=67,source_0d=zero_sources)
        fig=plt.figure(figsize=(WIDTH,67/25.4),layout='constrained');axes=[];central_metadata=[]
        for i,c in enumerate(center_cases):
            T=c['period'];use=(c['time']>=2600)&(c['time']<=2600+(T or 0)+1e-8)
            ax=fig.add_subplot(1,3,i+1,projection='3d');axes.append(ax);d=load(c['name'])
            vector=c['stokes'][use];fraction=(c['time'][use]-2600)/T if T else None
            endpoint_error=0.
            if T:
                ix,w=_lag_polynomial_weights(c['time'],2600+T,8);S_end=w@stokes(c['psi'][ix],axis=1).T;endpoint=S_end[1:]/S_end[0]
                endpoint_error=float(abs(np.dot(endpoint,endpoint)-1))
                if endpoint_error>.002:raise ValueError('Cycle endpoint needs denser sampling')
                vector=np.vstack((vector,endpoint));fraction=np.r_[fraction,1.]
            sphere(ax,vector,rf"$J={1000*d['config']['physical']['J']:g}$ $\mu$eV"+'\n'+c['label'],fixed=T is None,fractions=fraction)
            central_metadata.append({'run':c['name'],'coordinates_um':[0,0],'start_ps':2600.,'period_ps':T,'actual_last_sample_ps':float(c['time'][use][-1]),'display_end_ps':2600+T if T else 2600.,'endpoint_interpolation_unit_residual':endpoint_error,'recurrence_max':c['recurrence_max'],'interpolation_crosscheck_max':c['interpolation_crosscheck_max'],'classification':c['label']})
        fig.colorbar(plt.cm.ScalarMappable(norm=Normalize(0,1),cmap='viridis'),ax=axes,orientation='horizontal',fraction=.045,pad=.02,label=r'$(t-t_0)/T_c$ (periodic trajectories only)',ticks=[0,.5,1])
        periods=', '.join(f"{v['run']}: T={v['period_ps']:.6g}ps" for v in central_metadata if v['period_ps'])
        finish(fig,'N11_spot_Poincare','Local center(0,0) polarization of the same three2D runs as N13: J=.040,.065,.100meV; p=13.8, epsilon=.3, beta=.0119meV um^2, identical preparation. '+periods+'. Orange dot marks the start at2600ps; arrows show increasing time. Color is elapsed fraction of each periodic trajectory only. Fixed state is a point. The final cycle endpoint is evaluated by the validated local8-point polynomial in unnormalized Stokes, then divided by S0; no artificial closing segment or unit-vector normalization. No averaging or replacement observation point. Local Stokes recurrence and temporal interpolation cross-check pass; periods are confirmed returns, not asserted minimal periods. N8 is the starting snapshot of the running case.',height=67,center_cycles=central_metadata)
        with open(directory/'spot_J_diagnostic.json') as stream:scan2=json.load(stream)['cases']
        candidates={}
        for r in scan2:
            if r['beta']!=.0119 or r['L_um']!=256 or r['dt_ps']!=.025:continue
            if r['J_meV'] not in candidates or r['N']<candidates[r['J_meV']]['N']:candidates[r['J_meV']]=r
        fig,axes=plt.subplots(1,2,figsize=(WIDTH,68/25.4),layout='constrained');colors={'RUNNING':'C1','BOUNDED':'C2','LOCKED':'C0','UNRESOLVED':'.5'}
        for J,r in sorted(candidates.items()):
            for ax,key in zip(axes,('A_Z','phase_drift_rad_ps')):ax.plot(1000*J,r[key],'o',color=colors[r['regime']],ms=4)
        axes[0].set(ylabel=r'Integral imbalance amplitude, $A_Z$',title='(a) Population oscillations');axes[1].set(ylabel=r'$\langle\dot\Theta\rangle$ (rad ps$^{-1}$)',title='(b) Relative-phase drift')
        for ax in axes:ax.set_xlabel(r'Coherent coupling, $J$ ($\mu$eV)');ax.axhline(0,color='.65',lw=.6)
        fig.legend(handles=[Line2D([],[],marker='o',ls='',color=color,label=label) for label,color in [('Running','C1'),('Oscillating','C2'),('Fixed','C0'),('Unresolved','.5')]],loc='outside upper center',ncol=4,frameon=False)
        finish(fig,'N12_spot_J_scan','TE-TM spatial scan: p=13.8, epsilon=.3, beta=.0119meV um^2. A_Z is half peak-to-peak of the INTEGRAL population imbalance Z=(N+-N-)/(N++N-). Phase is Theta=-arg integral(S1+iS2)dA, not a relabelled central phase. Two adjacent late analysis windows, each at least max(200ps,5T); they end at3000ps except continued J=.055/.075 at4500ps. Integer-cycle means in accepted periodic windows; actual windows and statuses are recorded per point in spot_J_diagnostic.json. Color denotes phase family; gray is unresolved. Selected J=.040,.065,.100 cases have independently checked matching central classifications. No interpolation across unresolved points. Confirmed rotating and bounded examples bracket a possible transition in(.045,.060)meV; .055 remains locally phase-masked. Bounded-to-fixed transition is bracketed by .070 and .075meV. These sampled intervals are not precise bifurcation values.',height=68,scan_points=list(candidates.values()))
        fig,axes=plt.subplots(2,3,figsize=(WIDTH,96/25.4),layout='constrained')
        for col,c in enumerate(center_cases):
            duration=3*(c['period'] or max(v['period'] or 0 for v in center_cases));use=(c['time']>=2600)&(c['time']<=2600+duration)
            t=c['time'][use]-2600;phase=c['phase'][use];phase-=2*np.pi*np.floor((phase[0]+np.pi)/(2*np.pi))
            axes[0,col].plot(t,c['z'][use],lw=1);axes[1,col].plot(t,phase,lw=1,color='C1');J=load(c['name'])['config']['physical']['J']
            axes[0,col].set_title(c['label']+'\n'+rf'$J={J*1000:g}$ $\mu$eV',fontsize=9);axes[0,col].set_ylim(-1,1)
            if c['period'] is None:
                axes[1,col].set_ylim(float(np.mean(phase))-.05,float(np.mean(phase))+.05);axes[1,col].ticklabel_format(axis='y',style='plain',useOffset=False)
            axes[1,col].set_xlabel(r'$t-2600$ (ps)');axes[1,col].set_ylabel(r'$\theta(0,0,t)$ (rad)')
        axes[0,0].set_ylabel(r'Central imbalance, $z(0,0,t)$')
        finish(fig,'N13_spot_regime_dynamics','Central observables only at(0,0) for the same runs as N11: J=.040,.065,.100meV, all other parameters and seeds identical. z=(n+-n-)/(n++n-); theta=arg psi+-arg psi-. Three confirmed periods shown for each oscillating trajectory, a comparable duration for the stationary case. Phase is continuously unwrapped; only integer2pi branch offsets are applied. The fixed phase retains its actual value. Central classification is checked separately from whole-map stationarity/periodicity and does not imply the same local phase behavior everywhere.',height=96,center_cycles=central_metadata)
    # Keep the optional alpha2 candidate among diagnostics, not in the main set.
    reserve=ROOT/'results/diagnostics/report3/F3_cross_spin_density_comparison.png'
    manifest['reserve']={'N2':{'file':str(reserve.relative_to(ROOT)),'status':'DIAGNOSTIC_ONLY','caption_en':'Optional radial alpha2 comparison A0/A1; small differences lack an independent paired error bound. No large extra series is scheduled.'}}
    main_paths=[v for v in paths if v.parent==out]
    if len(main_paths)!=16 or set(manifest['figures'])-{'N6_spot_coupling_dynamics'}!=set(manifest['figure_order']):raise ValueError('Incomplete report figure set')
    # Remove superseded report-3 PNG prototypes only after the complete replacement exists.
    keep={p.name for p in main_paths}
    for path in out.glob('*.png'):
        if path.name not in keep:path.unlink()
    manifest['chapters']={'1': {'figures': ['P1_reservoir_switching']}, '2': {'figures': ['P2_homogeneous_pump_response', 'N1_spatial_component_densities'], 'reserve': 'N2'}, '3': {'figures': ['P3_homogeneous_densities', 'P4_homogeneous_relative_phase', 'P5_homogeneous_J_scan']}, '4': {'figures': ['N3_TE_TM_spot', 'N4_TE_TM_ring', 'N5_TE_TM_angular_response',  'N7_ring_angular_polarization','N12_spot_J_scan','N13_spot_regime_dynamics']}, '5': {'figures': ['N8_spot_Stokes', 'N9_ring_Stokes', 'N10_homogeneous_Poincare', 'N11_spot_Poincare']}}
    write_json(manifest_path,manifest)
    if animations:_report3_saved_gifs(out,load,manifest)
    return main_paths
