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
    fig.canvas.draw();renderer=fig.canvas.get_renderer();bounds=fig.bbox
    axes=list(fig.axes)
    for ax in list(axes):axes.extend(ax.child_axes)
    texts=list(fig.texts)
    for ax in axes:
        texts.extend([ax.title,ax._left_title,ax.xaxis.label,ax.yaxis.label,*ax.texts])
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


def save(fig,out,name):
    out=Path(out);out.mkdir(parents=True,exist_ok=True)
    height=FIGURE_HEIGHTS_MM.get(name,fig.get_size_inches()[1]*25.4)
    fig.set_size_inches(WIDTH,height/25.4);fig.set_dpi(300)
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
        else:fig.suptitle('Relative phase (different vertical scales); same time windows as Fig. 4',fontsize=8.5)
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
    frequency=np.array([-r['mean_phase_drift'] if r.get('mean_phase_drift') is not None else np.nan for r in rows])
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
    axes[1].set_title('(b) Mean frequency difference',loc='left')
    axes[1].set_ylabel(r'$\overline{\omega_+-\omega_-}$ (ps$^{-1}$)')
    axes[1].ticklabel_format(axis='y',style='sci',scilimits=(-2,-2),useMathText=True)
    axes[1].axhline(0,color='.6',lw=.7,zorder=0)
    handles=[Line2D([],[],ls='',marker=m,color=c,label=rf'$J={j:g}$ $\mu$eV',ms=5)
             for j,m,c in ((5,'o','C1'),(6.5,'s','C2'),(10,'D','C3'))]
    fig.legend(handles=handles,loc='outside upper center',ncol=3,frameon=False,
               title='Examples shown in Figs. 4 and 5',title_fontsize=8.5)
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
        path=ROOT/'docs'/filename;text=path.read_text();start=f'<!-- {tag}_START -->';end=f'<!-- {tag}_END -->'
        if start not in text or end not in text:raise ValueError('Missing documentation markers: '+filename)
        before,rest=text.split(start,1);_,after=rest.split(end,1)
        path.write_text(before+start+'\n'+content+'\n'+end+after)
    _update_latex_spatial_results(report)


def _update_latex_spatial_results(report):
    """Update both report languages from the same measured N3 results."""
    import re,shutil,subprocess
    path=ROOT/'latex/report.tex'
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
    path=ROOT/'docs/results_overview.md';text=path.read_text();tag='REPORT3_PILOTS'
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


def make_report3_figures(*,animations=True):
    """CPU-only rendering. No integration, PyTorch import, or report text edits."""
    from model import stokes
    from numerics import write_json
    from scipy.stats import binned_statistic
    from PIL import Image
    import hashlib
    directory=DATA/'report3';out=ROOT/'results/figures/report3';out.mkdir(parents=True,exist_ok=True)
    diagnostic=ROOT/'results/diagnostics/report3'
    manifest={'method':'CPU postprocessing of saved arrays; no field integration or temporal interpolation',
              'PNG_width_mm':160,'PNG_dpi':300,'mask_relative_S0':.001,'figures':{},'animations':{},
              'limitations':['T0 ring polarization still evolves at6000 ps','Small alpha2 contribution has no paired error bound','Only two modes are verified in the new0D sample; no2D J scan has been performed']}
    prior=directory/'figure_manifest.json'
    if prior.exists() and not animations:
        manifest['animations']=json.loads(prior.read_text()).get('animations',{})
    queue_file=directory/'experiment_queue.json'
    queue_state=json.loads(queue_file.read_text()) if queue_file.exists() else {}
    def c1_status(geometry):
        records=queue_state.get('experiments',{})
        return 'READY' if all(records.get('C1_'+geometry+'_'+kind,{}).get('convergence',{}).get('status')=='PASS' for kind in ('dt','grid')) else 'PENDING_C1_CHECKS'
    cache={};paths=[]
    reviews=json.loads((directory/'report3_cycle_review.json').read_text())['cases']
    def load(name):
        if name not in cache:
            with np.load(directory/(name+'.npz'),allow_pickle=False) as z:
                cache[name]={k:z[k].copy() for k in ('psi','x','y','history','features','mean_maps','mean_interval_ps')}
                cache[name]['config']=json.loads(str(z['config_json']))
                cache[name]['summary']=json.loads(str(z['summary_json']))
            cache[name]['name']=name
        return cache[name]
    def record(key,path,caption,members=(),status='READY',**extra):
        manifest['figures'][key]={'file':str(path.relative_to(ROOT)),'caption_en':caption,'status':status,
             'sources':[{'run':d['name'],'time_ps':float(d['history'][-1,0]),'field_sha256':hashlib.sha256(np.ascontiguousarray(d['psi']).tobytes()).hexdigest(),'config':d['config']} for d in members],**extra}
        write_json(directory/'figure_manifest.json',manifest);paths.append(path);print('FIGURE',path.name,status,flush=True)
    def finish(fig,key,caption,members=(),status='READY',height=72,where=out,**extra):
        fig.set_size_inches(WIDTH,height/25.4)
        path=save(fig,where,key);record(key,path,caption,members,status,**extra);return path
    def extent(x):
        dx=x[1]-x[0];return [x[0]-dx/2,x[-1]+dx/2]*2
    def draw(ax,image,d,title,*,polar=False,vmax=None,span=25):
        x=d['x'];ix=np.flatnonzero(abs(x)<=span);im=image[np.ix_(ix,ix)]
        if polar:
            S0=abs(d['psi'])**2;S0=S0.sum(0)
            im=np.ma.masked_where(S0[np.ix_(ix,ix)]<.001*S0.max(),im)
        cmap=plt.get_cmap('RdBu_r' if polar else 'viridis').copy();cmap.set_bad('#ececec')
        artist=ax.imshow(im,origin='lower',extent=extent(x[ix]),interpolation='nearest',cmap=cmap,
                          vmin=-1 if polar else 0,vmax=1 if polar else vmax)
        cfg=d['config'];g=make_grid(cfg);pump=pump_profile(cfg,g)
        ax.contour(x[ix],x[ix],pump[np.ix_(ix,ix)],levels=[.5],colors='.25' if polar else 'white',linewidths=.55)
        ax.set(title=title,xlabel=r'$x$ ($\mu$m)',ylabel=r'$y$ ($\mu$m)',aspect='equal',xticks=[-20,0,20],yticks=[-20,0,20])
        ax.grid(False);return artist
    def parameters_caption(d):
        c=d['config'];p=parameters(c)
        return f"{d['name']}: p={c['pump']['P0']:g}, epsilon={c['pump']['epsilon']:g}, alpha2={p['alpha2']:g} meV um^2, J={p['J']:g} meV, beta={p['beta']:g} meV um^2; N={c['space']['N']}, L={c['space']['L']:g} um, dt={c['time']['dt']:g} ps. "
    def density_panel(geometry,name,key,provisional=False):
        d=load(name);n=abs(d['psi'])**2;S0=n.sum(0);s3=(n[0]-n[1])/np.maximum(S0,1e-300)
        fig,axs=plt.subplots(1,3,layout='constrained',figsize=(WIDTH,2.8))
        for k in (0,1):im=draw(axs[k],n[k],d,f'({chr(97+k)}) '+('$n_+$' if k==0 else '$n_-$'),vmax=float(n.max()))
        fig.colorbar(im,ax=axs[:2],fraction=.035,pad=.02,label=r'Density ($\mu$m$^{-2}$)')
        im=draw(axs[2],s3,d,r'(c) Circular $s_3$',polar=True)
        fig.colorbar(im,ax=axs[2],fraction=.065,pad=.025,ticks=[-1,0,1])
        for ax in axs[1:]:ax.set_ylabel('');ax.tick_params(labelleft=False)
        t=d['history'][-1,0]
        pop=n.sum((-2,-1))*(d['x'][1]-d['x'][0])**2
        caption=f'Densities and circular polarization under {geometry} pumping with TE-TM coupling, at t={t:g} ps. '+parameters_caption(d)
        caption+=f'N+={pop[0]:.6g}, N-={pop[1]:.6g}, Z={(pop[0]-pop[1])/pop.sum():.6g}. Both densities share one absolute scale. Thin contours: half-maximum pump; polarization pixels below 0.001 of peak S0 are masked. '
        caption+=('Densities are stationary within the stated tolerance, while full polarization continues to drift. This is a finite-time snapshot, not a phase-locked state.' if provisional else 'Density and polarization stationarity were checked on the saved late windows.')
        finish(fig,key,caption,[d],'READY_FINITE_TIME' if provisional else 'READY',height=65)
    with plt.rc_context(STYLE):
        # Reuse the matched scalar switching calculation, without recomputation.
        path=report_figure1(out);newpath=out/'F1_reservoir_switching.png';path.replace(newpath)
        record('F1_reservoir_switching',newpath,'Condensate density after pump turn-on and turn-off in the matched one- and two-reservoir models. PA=1.5 Pth; identical seed and initially empty reservoirs. Time in (b) is measured from pump turn-off at1000 ps. The dotted level is the common stationary density. Saved F1_M0/F1_R1 data from the preceding report are reused.')
        fig,axs=plt.subplots(2,2,figsize=(WIDTH,4.5),layout='constrained')
        members=[]
        for row,(geometry,name) in enumerate((('Gaussian spot','production_A1_spot'),('Annular pump','matched_A1_ring'))):
            d=load(name);members.append(d);n=abs(d['psi'])**2
            for k in (0,1):im=draw(axs[row,k],n[k],d,f'({chr(97+row*2+k)}) '+geometry+(': $n_+$' if k==0 else ': $n_-$'),vmax=float(n.max()))
            axs[row,1].set_ylabel('');axs[row,1].tick_params(labelleft=False)
            fig.colorbar(im,ax=axs[row],fraction=.035,pad=.02,label=r'Density ($\mu$m$^{-2}$)')
        caption='Circular-component density profiles with pump imbalance epsilon=0.3, J=beta=0. '+''.join(parameters_caption(d) for d in members)
        for d in members:
            n=abs(d['psi'])**2;pop=n.sum((-2,-1))*(d['x'][1]-d['x'][0])**2
            caption+=f"{d['name']}: snapshot at{d['history'][-1,0]:g} ps, N+={pop[0]:.6g}, N-={pop[1]:.6g}, Z={(pop[0]-pop[1])/pop.sum():.6g}. "
        caption+='A common absolute density scale is used within each geometry. Density stationarity does not imply fixed relative phase. Thin contours mark half-maximum pump; w=d=2.5 um, ring radius10 um.'
        finish(fig,'F2_circular_component_densities',caption,members,height=112)
        fig,axs=plt.subplots(1,2,figsize=(WIDTH,2.8),layout='constrained');members=[]
        for ax,geom in zip(axs,('spot','ring')):
            for case,ls in (('A0','-'),('A1','--')):
                d=load(('production_' if geom=='spot' else 'matched_')+case+'_'+geom);members.append(d)
                radius=np.hypot(d['x'][None,:],d['y'][:,None]);edges=np.linspace(0,30,121)
                for j in (0,1):
                    radial=binned_statistic(radius.ravel(),(abs(d['psi'][j])**2).ravel(),statistic='mean',bins=edges)[0]
                    ax.plot((edges[:-1]+edges[1:])/2,radial,color=f'C{j}',ls=ls,label=case+(': $n_+$' if j==0 else ': $n_-$'))
            ax.set(title='Gaussian spot' if geom=='spot' else 'Annular pump',xlabel=r'Radius ($\mu$m)',ylabel=r'Density ($\mu$m$^{-2}$)',xlim=(0,25))
        axs[0].legend(frameon=False,ncol=2,fontsize=7)
        finish(fig,'F3_cross_spin_density_comparison','Radial density profiles for alpha2=0 (A0) and alpha2=-0.00024 meV um^2 (A1); all other parameters and preparation are held fixed within each geometry. Small differences have no independent paired convergence bound and are not claimed as a resolved alpha2 effect. Annular snapshots may retain evolving polarization.',members,status='DIAGNOSTIC_ONLY',height=72,where=diagnostic)
        density_panel('Gaussian-spot','production_T0_spot','F4_TE_TM_spot')
        density_panel('annular','matched_T0_ring','F5_TE_TM_ring',provisional=load('matched_T0_ring')['summary']['analysis']['status']!='STATIONARY')
        fig,axs=plt.subplots(1,2,figsize=(WIDTH,2.9),layout='constrained');members=[];windows={}
        for ax,geom in zip(axs,('spot','ring')):
            names=['production_B0_spot','report3_T0_spot_beta_0.5','production_T0_spot','report3_T0_spot_beta_2'] if geom=='spot' else ['matched_B0_ring','report3_T0_ring_beta_0.5','matched_T0_ring','report3_T0_ring_beta_2']
            means=[];low=[];high=[]
            for name in names:
                d=load(name);members.append(d);f=d['features'];f=f[f[:,0]>=f[-1,0]-400];v=2*np.hypot(f[:,10],f[:,11]);mean=float(np.trapezoid(v,f[:,0])/(f[-1,0]-f[0,0]));means.append(mean);low.append(mean-v.min());high.append(v.max()-mean);windows[name]=[float(f[0,0]),float(f[-1,0])]
            ax.errorbar([0,.5,1,2],means,yerr=[low,high],fmt='o-' if geom=='spot' else 'o',ms=4,lw=1,elinewidth=.8,capsize=2,color='C0' if geom=='spot' else 'C1')
            ax.set(title='Gaussian spot' if geom=='spot' else 'Annular pump',xlabel=r'TE-TM coefficient, $\beta/\beta_{\rm ref}$',ylabel=r'Second angular harmonic, $Q_2$',xticks=[0,.5,1,2],xlim=(-.1,2.1));ax.set_ylim(bottom=-.005)
            if geom=='ring':
                ax.plot(1.,means[2],'o',ms=5,mfc='white',mec='C1',zorder=4,label='Evolving reference')
                ax.annotate('Evolving reference',xy=(1.,means[2]),xytext=(8,-18),textcoords='offset points',fontsize=7)
        finish(fig,'F6_TE_TM_angular_response','Second angular harmonic Q2=2|C2| versus beta/beta_ref. Markers are time means and vertical bars show the temporal range, not statistical or numerical uncertainty. ROI: r=5-15 um for the spot, r=5-7.5 um for the ring. Each explicitly recorded late window is400 ps; ring T0 is still evolving, so no established monotonic ring response is claimed. The controlled initial seed is shared within each geometry; beta_ref=.0119 meV um^2. The hollow ring marker denotes the evolving beta_ref state at6000 ps. All points use the same Cartesian ROI quadrature; temporal bars do not include its discretization uncertainty. The independent C1 field-convergence controls passed, without providing an error bound for all beta variants.',members,status='READY_FINITE_TIME',height=76,windows_ps=windows)
        fig,axs=plt.subplots(2,2,figsize=(WIDTH,3.5),layout='constrained');sources=[]
        for col,J in enumerate((.005,.01)):
            source=directory/f'report3_0d_J_{J:.8f}.npz'
            with np.load(source,allow_pickle=False) as z:t=z['time'];psi=z['psi'];summary=json.loads(str(z['summary_json']))
            keep=t>=t[-1]-1200;t=t[keep];psi=psi[keep];n=abs(psi)**2;phase=np.unwrap(np.angle(psi[:,0]*psi[:,1].conj()));phase-=phase[0]
            axs[0,col].plot(t-t[0],(n[:,0]-n[:,1])/n.sum(1),color='C0');axs[1,col].plot(t-t[0],phase,color='C1')
            axs[0,col].set_title(('(a) Running phase' if col==0 else '(b) Locked phase')+f'; $J={1000*J:g}$ '+r'$\mu$eV',loc='left')
            axs[0,col].set_ylabel(r'Spin imbalance, $z$');axs[1,col].set_ylabel(r'Phase change, $\theta-\theta_0$ (rad)');axs[1,col].set_xlabel('Time in late window (ps)')
            axs[0,col].set_ylim(-1,1);axs[1,col].set_ylim(-24,1);sources.append({'file':str(source.relative_to(ROOT)),'display_interval_ps':[float(t[0]),float(t[-1])],'status':summary['status']})
        finish(fig,'F7_homogeneous_phase_dynamics','Spin imbalance z and change of unwrapped relative phase theta=arg(psi+)-arg(psi-) for independent homogeneous runs at J=.005 and .010 meV, p=2, epsilon=.3, beta=0. Each panel displays the final1200 ps, measured from the start of its own display window; theta0 is the phase at that start. Classification uses two validated windows, not only the displayed segment. Both components remain nonzero. Running and locked phase states survive tighter independent tolerances.',height=90,source_0d=sources)
        for geometry in ('spot','ring'):
            fig,ax=plt.subplots(figsize=(WIDTH,2.8),layout='constrained');members=[];windows={};quadrature_errors={}
            coupling_labels={'A1':'No coherent coupling','T1':'TE-TM only','J1':'J only','C1':'J + TE-TM'}
            for case,color,ls in (('A1','0.35',':'),('T1','C0','-'),('J1','C2','--'),('C1','C1','-')):
                name=('production_'+case+'_spot') if geometry=='spot' else ('charge_J1_seeded_c128_L256_t3600' if case=='J1' else 'matched_'+case+'_ring')
                d=load(name);members.append(d);f=d['features'];length=540. if geometry=='spot' else 600.
                end=float(f[-1,0]);begin=end-length;f=f[f[:,0]>=begin]
                windows[name]=[begin,end];q=2*np.hypot(f[:,10],f[:,11])
                if geometry=='spot':
                    ax.plot(f[:,0]-begin,2*f[:,10],label=coupling_labels[case],color=color,ls=ls,alpha=.85)
                else:
                    angles,angular=report3_polar_profile(d['psi'],d['x'])
                    _,cross=report3_polar_profile(d['psi'],d['x'],order=5)
                    quadrature_errors[name]=float(np.max(abs(angular-cross)))
                    if quadrature_errors[name]>.001:raise ValueError('Polar interpolation requires refinement: '+name)
                    ax.plot(angles,angular,label=coupling_labels[case],color=color,ls=ls,alpha=.85)
            if geometry=='spot':
                ax.set(xlabel='Time in late window (ps)',ylabel=r'Cosine angular component, $2\,\mathrm{Re}\,C_2$',xlim=(0,length),ylim=(-.17,.17))
            else:
                ax.set(xlabel=r'Polar angle, $\varphi$ (degrees)',ylabel=r'Circular polarization, $s_3(\varphi)$',xlim=(-180,180),xticks=[-180,-90,0,90,180])
                ax.margins(y=.15)
            if geometry=='spot':fig.legend(*ax.get_legend_handles_labels(),loc='outside upper center',frameon=False,ncol=2)
            else:ax.legend(loc='upper center',bbox_to_anchor=(.5,1.28),frameon=False,ncol=2)
            caption='Angular circular-polarization response under '+('Gaussian' if geometry=='spot' else 'annular')+' pumping, epsilon=.3. A1: J=beta=0; T1: J=0,beta_ref; J1: J_ref,beta=0; C1: J_ref,beta_ref. '
            caption+=('The cosine angular component2 Re C2 retains the oscillation hidden by the nearly constant modulus Q2. All runs share interval2460-3000 ps; fixed ROI5-15 um. Full-map periodicity is verified. ' if geometry=='spot' else 'Angular s3 is the ratio of integrated S3 and S0 in each of72 polar sectors over r=5-7.5 um. The same Gauss quadrature (32 radial and8 angular nodes per sector) is applied to cubic-interpolated complex fields; a fifth-order cross-check is recorded. This removes unequal Cartesian pixel counts without angular filtering. A1/T1 have periodic observables; J1/C1 are stationary. Each snapshot uses its explicitly recorded physical time and is not phase-aligned across different parameter sets. The four starts share the same explicit seed. ')
            finish(fig,'F8_'+geometry+'_coupling_comparison',caption,members,status=c1_status('spot') if geometry=='spot' else 'READY_PERIODIC_AND_STATIONARY',height=82,windows_ps=windows,polar_order3_vs5_max=quadrature_errors)
        for geom,name,key in [('spot','production_C1_spot','F9_spot_Stokes'),('ring','matched_C1_ring','F10_ring_Stokes')]:
            d=load(name);S=stokes(d['psi']);values=S[1:]/np.maximum(S[0],1e-300)
            fig,axs=plt.subplots(1,3,figsize=(WIDTH,2.8),layout='constrained')
            for j,ax in enumerate(axs):im=draw(ax,values[j],d,f'({chr(97+j)}) '+rf'$s_{j+1}$',polar=True)
            for ax in axs[1:]:ax.set_ylabel('');ax.tick_params(labelleft=False)
            fig.colorbar(im,ax=axs,fraction=.025,pad=.02,ticks=[-1,0,1],label='Normalized Stokes parameter')
            caption=('Gaussian-spot' if geom=='spot' else 'Annular-pump')+f" C1 polarization at t={d['history'][-1,0]:g} ps. "+parameters_caption(d)
            caption+='S1=2Re(conj(psi+)*psi-), S2=2Im(conj(psi+)*psi-), S3=n+-n-; si=Si/S0. One common [-1,1] scale; low-intensity pixels are masked. '
            caption+=('Instantaneous phase of a verified periodic state; the accompanying GIF shows its evolution, so this is not a time mean. ' if geom=='spot' else 'Snapshot of the verified stationary polarization. ')
            caption+=('Independent C1 space/time controls passed.' if c1_status(geom)=='READY' else 'Final C1 space/time controls are pending.')
            finish(fig,key,caption,[d],status=c1_status(geom),height=65)
    if animations:
        manifest['animations'].update(_report3_saved_gifs(out,load,manifest))
    manifest['chapters']={
        '1':{'status':'READY','figures':['F1_reservoir_switching']},
        '2':{'status':'DENSITY_FIGURE_READY_ALPHA2_UNRESOLVED','figures':['F2_circular_component_densities'],'diagnostic':'F3_cross_spin_density_comparison'},
        '3':{'status':'READY_WITH_DYNAMIC_RING','figures':['F4_TE_TM_spot','F5_TE_TM_ring','F6_TE_TM_angular_response']},
        '4':{'status':'READY_FOR_SAMPLED_CASES_NO_2D_J_SCAN','figures':['F7_homogeneous_phase_dynamics','F8_spot_coupling_comparison','F8_ring_coupling_comparison']},
        '5':{'status':'READY_C1_CONVERGENCE_VERIFIED','figures':['F9_spot_Stokes','F10_ring_Stokes']}}
    write_json(directory/'figure_manifest.json',manifest)
    return paths


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
        # Prefer the forthcoming time-step control for full spot history.
        if case=='C1' and geom=='spot':
            candidate=ROOT/'results/runs/report3/check_C1_spot_full_dt0125'
            if (candidate/'summary.json').exists():
                summary=json.loads((candidate/'summary.json').read_text())
                if summary.get('run_status')=='RUN_COMPLETED' and summary.get('time_ps',0)>=3000:
                    with np.load(candidate/'checkpoint.npz') as z:ts=z['probe_times'];has_zero=np.any(np.isclose(ts,0,rtol=0,atol=1e-8))
                    if has_zero:name='check_C1_spot_full_dt0125'
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
        with np.load(run/'checkpoint.npz',allow_pickle=False) as z:times=z['probe_times'].copy()
        maps=np.load(run/'probe_maps.npy',mmap_mode='r',allow_pickle=False);local=[];N=c['space']['N'];dx=c['space']['L']/N
        for x,y in ((0,0),(2,0),(-2,0),(0,2),(0,-2)):
            point=[]
            for left,right in intervals:
                ids=np.flatnonzero((times>=left-1e-8)&(times<=right+1e-8));ids=ids[np.argsort(times[ids])]
                v=np.asarray(maps[ids,:,round(N/2+y/dx),round(N/2+x/dx)])
                plus=v[:,0]+1j*v[:,1];minus=v[:,2]+1j*v[:,3];arg=np.angle(plus*minus.conj());pp=np.unwrap(arg)
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
        axes[2,0].set_ylabel('Mean phase drift\n'+r'(rad ps$^{-1}$)')
        for ax in axes[1]:ax.set_ylim(-.025,.77)
        regimes=[Line2D([],[],color=color,lw=2,label=label) for key,(color,label) in styles.items() if any(r['regime']==key for r in rows)]
        fig.legend(handles=regimes,loc='outside upper center',ncol=2,frameon=False,title='Diagnostic 2D scan: Gaussian spot')
        handles=[Line2D([],[],color='.25',mfc='none',ls='',marker=m,label=label,ms=5) for key,(m,label) in symbols.items() if any(resolution(r)==key for r in rows)]
        fig.legend(handles=handles,loc='outside lower center',ncol=2,fontsize=7,frameon=False,title=r'Numerical controls: $L$ in $\mu$m, $\Delta t$ in ps; complex128')
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
