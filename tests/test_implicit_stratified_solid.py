"""Tall columns, solid-wall mass closure, and clamped face-work energy (#283).

Every face-work energy check measures E + P (tall.radial_exact(): gravity-work: face always
books the corrected-PE work), the energy that work conserves (P the corrected PE of
docs/derivations/curved-gravity-work-weight.md sec 7), on the final state; on a
gnomonic-equiangle face block P also carries the centroid term of sec 12."""
import json
import math
import sys
import tempfile
from pathlib import Path
import torch
import yaml
import test_implicit_face_work_operator as tall
import test_gravity_work_fixer as box
import snapy
from snapy import MeshBlock, MeshBlockOptions, kIDN, kIPR, kIV1
torch.set_default_dtype(torch.float64)
device = sys.argv[1] if len(sys.argv) > 1 else 'cpu'

def create(cfg):
    with tempfile.NamedTemporaryFile('w', suffix='.yaml', dir=Path(__file__).parent) as f:
        yaml.safe_dump(cfg, f)
        f.flush()
        options = MeshBlockOptions.from_yaml(f.name)
        options.layout().device(device.split(':')[0])
        b = MeshBlock(options)
    b.to(torch.device(device), torch.float64)
    return b

def tall_run(nz, heights, scheme, work, steps=1, dt=None, fixer=False):
    H = tall.RD * tall.T0 / tall.GRAV
    dz = H * heights / nz
    cfg = tall.config(scheme)
    cfg['geometry']['cells']['nx1'] = nz
    cfg['geometry']['bounds']['x1max'] = dz * nz
    cfg['forcing']['const-gravity'].update({'gravity-work': work, 'gravity-work-fixer': fixer})
    b = create(cfg)
    ng = 3
    z = (torch.arange(nz) + 0.5) * dz
    p = tall.PS * torch.exp(-z / H)
    col = torch.zeros(kIPR + 1, 1, 1, nz)
    col[kIDN, 0, 0] = p / (tall.RD * tall.T0)
    col[kIPR, 0, 0] = p
    (wb, _, _) = snapy.balance_column(col, torch.full((nz,), dz), tall.GRAV)
    w = dict(b.named_buffers())['hydro.D'].clone().zero_()
    for c in (kIDN, kIPR):
        w[c][..., ng:ng + nz] = wb[c, 0, 0].to(w)
        w[c][..., :ng] = w[c][..., ng:ng + 1]
        w[c][..., ng + nz:] = w[c][..., ng + nz - 1:ng + nz]
    (v, _) = b.initialize({'hydro_w': w})
    sl = (Ellipsis, slice(ng, ng + tall.NX2), slice(ng, ng + nz))
    u0 = v['hydro_u'][sl].clone()

    m0 = u0[kIDN].sum().item()
    phi = tall.GRAV * z.to(u0)
    exact = work == 'face' and tall.radial_exact()
    x1f = torch.arange(nz + 1) * dz

    def energy(u):
        e = u[kIPR] + phi * u[kIDN]
        if exact:
            e = e + tall.corrected_pe(u[kIDN], x1f, z, -tall.GRAV)
        return e.sum().item()

    e0 = energy(u0)
    dt = dt or 657 * dz / math.sqrt(tall.GAMMA * tall.RD * tall.T0)
    out = {'nz': nz, 'H': heights, 'scheme': scheme, 'work': work, 'fixer': fixer, 'dt': dt, 'device': device}
    wmax = 0.0
    redos = 0
    completed = 0
    redo_status = 0
    rolled_back = None
    for n in range(steps):
        previous = v['hydro_u'][sl].clone()
        for stage in range(len(b.module('intg').stages)):
            b.forward(v, dt, stage)
        redo_status = b.check_redo(v)
        if redo_status:
            redos += 1
            rolled_back = bool(torch.equal(v['hydro_u'][sl], previous))
            break
        u = v['hydro_u'][sl]
        if not torch.isfinite(u).all():
            break
        wmax = max(wmax, (u[kIV1] / u[kIDN]).abs().max().item())
        completed = n + 1
    u = v['hydro_u'][sl]
    rho = u[kIDN]
    vel = u[kIV1] / rho
    temp = (tall.GAMMA - 1) * (u[kIPR] - 0.5 * rho * vel ** 2) / (rho * tall.RD)
    out['epe_drift'] = energy(u) / e0 - 1
    out.update(steps=completed, attempted_steps=n + 1, redos=redos, redo_status=redo_status, rolled_back=rolled_back, finite=bool(torch.isfinite(u).all()), mass=rho.sum().item() / m0 - 1, top_rho=rho[..., -1].mean().item() / u0[kIDN, ..., -1].mean().item() - 1, top_T=temp[..., -1].mean().item(), wmax=wmax)
    buf = dict(b.named_buffers())
    out['clamp'] = next((v for (k, v) in buf.items() if k.endswith('.dry_clamp_step'))).item()
    print(json.dumps(out), flush=True)
    return out

def solid_run(scheme, work, fixer, placement='top', strided=False):
    cfg = box.config({'gravity-work': work, 'gravity-work-fixer': fixer}, scheme=scheme)
    ng = 3
    nz = box.NZ
    dz = box.LZ / nz
    z = (torch.arange(nz) + 0.5) * dz
    T = box.TS - box.GRAV * z / box.CP
    p = box.PS * (T / box.TS) ** (box.CP / box.RD)
    col = torch.zeros(kIPR + 1, 1, 1, nz)
    col[kIDN, 0, 0] = p / (box.RD * T)
    col[kIPR, 0, 0] = p
    (wb, _, _) = snapy.balance_column(col, torch.full((nz,), dz), box.GRAV)
    start = {'top': nz - 4, 'middle': nz // 2 - 2, 'bottom': 0}[placement]
    cfg['boundary-condition']['internal'] = {'solid-density': float(wb[kIDN, 0, 0, start]), 'solid-pressure': float(wb[kIPR, 0, 0, start])}
    if strided:
        cfg['geometry']['cells']['nx2'] = nz
        cfg['geometry']['bounds']['x2max'] = float(nz)
    b = create(cfg)
    w = dict(b.named_buffers())['hydro.D'].clone().zero_()
    for c in (kIDN, kIPR):
        w[c][..., ng:ng + nz] = wb[c, 0, 0].to(w)
        w[c][..., :ng] = w[c][..., ng:ng + 1]
        w[c][..., ng + nz:] = w[c][..., ng + nz - 1:ng + nz]
    w[kIV1][..., ng:ng + nz] = box.MACH * math.sqrt(box.GAMMA * box.RD * box.TS) * torch.sin(math.pi * z / box.LZ).to(w)
    solid = torch.zeros_like(w[kIDN], dtype=torch.bool)
    if strided:
        solid = solid.transpose(-1, -2)
    solid[..., ng + start:ng + start + 4] = True
    w[kIV1][solid] = 0
    for (c, key) in ((kIDN, 'solid-density'), (kIPR, 'solid-pressure')):
        w[c][solid] = cfg['boundary-condition']['internal'][key]
    (v, _) = b.initialize({'hydro_w': w, 'solid': solid})
    spans = [(0, start), (start + 4, nz)]
    spans = [(lo, hi) for (lo, hi) in spans if hi > lo]
    initial = [v['hydro_u'][kIDN, ..., ng + lo:ng + hi].sum().item() for (lo, hi) in spans]
    dt = b.max_time_step(v)
    for n in range(20):
        for stage in range(len(b.module('intg').stages)):
            b.forward(v, dt, stage)
    u = v['hydro_u']
    result = {'solid': True, 'scheme': scheme, 'work': work, 'fixer': fixer, 'dt': dt, 'mass': max((abs(u[kIDN, ..., ng + lo:ng + hi].sum().item() / m - 1) for ((lo, hi), m) in zip(spans, initial))), 'placement': placement, 'finite': bool(torch.isfinite(u).all()), 'device': device}
    print(json.dumps(result), flush=True)
    return result

def clamp_energy(scheme, work):
    cfg = tall.config(scheme)
    cfg['forcing']['const-gravity']['gravity-work'] = work
    cfg['geometry']['cells'].update(nx1=8, nx2=1, nx3=1)
    cfg['geometry']['bounds'].update(x1max=8.0, x2max=1.0, x3max=1.0)
    cfg['forcing']['const-gravity']['grav1'] = -1.0
    b = create(cfg)
    w = b.buffer('hydro.D').clone().zero_()
    w[kIDN] = 1.0
    w[kIPR] = 1.0
    du = torch.zeros_like(w)
    du[kIV1, ..., 6] = 100.0
    b.module('hydro.icorr').forward(du, w, torch.full_like(w[kIDN], 1.4), 1.0)
    sl = (Ellipsis, slice(3, 11))
    phi = (torch.arange(8) + 0.5).to(w)
    defect = (du[kIPR][sl] + phi * du[kIDN][sl]).sum().item()
    if work == 'face' and tall.radial_exact():
        defect += tall.corrected_pe(du[kIDN][sl], torch.arange(9.), phi, -1.).sum().item()
    raw = b.buffer('hydro.icorr.delta').reshape(8, 5 if scheme == 9 else 3)
    raw_epe = (raw[:, -1] + phi * raw[:, 0]).sum().item()
    error = defect - (raw_epe if work == 'cell' else 0.0)
    print(json.dumps({'clamp_scheme': scheme, 'work': work,
                      'energy_defect': defect, 'raw_epe': raw_epe,
                      'redistribution_error': error}), flush=True)
    return error

def curved_energy(scheme, geometry, work):
    cfg=box.config({'gravity-work':'cell','gravity-work-fixer':False},scheme=scheme)
    cfg['geometry'].update(type=geometry)
    cfg['geometry']['cells'].update(nx1=8,nx2=6,nx3=6)
    if geometry=='gnomonic-equiangle':
        cfg['geometry']['bounds']={'x1min':10.,'x1max':18.,'x2min_pi':-.25,'x2max_pi':.25,'x3min_pi':-.25,'x3max_pi':.25}
    else:
        cfg['geometry']['bounds']={'x1min':10.,'x1max':18.,'x2min':.5,'x2max':2.5,'x3min':0.,'x3max':6.}
    cfg['forcing']['const-gravity']['grav1']=-1.
    cfg['forcing']['const-gravity']['gravity-work']=work
    b=create(cfg)
    w=b.buffer('hydro.D').clone().zero_()
    w[kIDN]=1.
    w[kIPR]=10.
    du=torch.zeros_like(w)
    du[kIPR,...,7]=1.
    du0=du.clone()
    dt=.1
    b.module('hydro.icorr').forward(du,w,torch.full_like(w[kIDN],1.4),dt)
    coord=b.module('coord')
    z=b.buffer('coord.x1v')[3:11]
    sl=(slice(3,9),slice(3,9),slice(3,11))
    if geometry=='gnomonic-equiangle':
        vol=coord.cell_volume()[sl]
        faces=coord.face_area1()[3:9,3:9,4:11]
    else:
        rf=b.buffer('coord.x1f')[3:12]
        theta=b.buffer('coord.x2f')[3:10]
        az=b.buffer('coord.x3f')[3:10]
        angular=(az[1:]-az[:-1]).unsqueeze(-1)*(theta[:-1].cos()-theta[1:].cos()).unsqueeze(0)
        vol=angular.unsqueeze(-1)*(rf[1:].pow(3)-rf[:-1].pow(3))/3.
        faces=angular.unsqueeze(-1)*rf[1:-1].square()
    raw=b.buffer('hydro.icorr.delta').reshape(6,6,8,5 if scheme==9 else 3)
    rho=raw[...,0]; momentum=raw[...,1]; energy=raw[...,-1]
    observed=((energy+z*rho-du0[kIPR][sl])*vol).sum()
    adv=.5*(momentum[...,:-1]+momentum[...,1:])
    expected=0. if work=='face' else -dt*((momentum*vol).sum()-(faces*(z[1:]-z[:-1])*adv).sum())
    final=((du[kIPR][sl]+z*du[kIDN][sl]-du0[kIPR][sl])*vol).sum()
    error=max((observed-expected).abs().item(),(final-expected).abs().item())
    if work=='face' and tall.radial_exact():
        # E + P on the final state: the solve books part of the corrected-PE work,
        # the post-solve term the rest, so the raw solve closes neither form alone
        final=final+(tall.corrected_pe(du[kIDN][sl],b.buffer('coord.x1f')[3:12],z,-1.,True,
                                       geometry=='gnomonic-equiangle')*vol).sum()
        error=(final-expected).abs().item()
    scale=(du0[kIPR][sl]*vol).abs().sum().item()
    out={'device':device,'geometry':geometry,'scheme':scheme,'work':work,'observed':observed.item(),'final':final.item(),'expected':float(expected),'error':error,'relative':error/scale,'finite':bool(torch.isfinite(du).all())}
    print(json.dumps(out),flush=True)
    return out

def coarse_restart():
    cfg = tall.config(0)
    cfg['forcing']['const-gravity'].update(
        {'gravity-work': 'cell', 'gravity-work-fixer': False})
    cfg['geometry']['bounds']['x1max'] = 40 * tall.RD * tall.T0 / tall.GRAV
    b = create(cfg)
    w = b.buffer('hydro.D').clone().zero_()
    w[kIDN] = 1
    w[kIPR] = tall.RD * tall.T0
    v, _ = b.initialize({'hydro_w': w})

    class Restart(torch.nn.Module):
        def forward(self, x):
            return x

    state = Restart()
    for name, value in {
        'hydro_u': v['hydro_u'],
        'last_time': torch.tensor(0.),
        'last_cycle': torch.tensor(0),
        'file_number': torch.empty(0, dtype=torch.int64),
        'next_time': torch.empty(0),
    }.items():
        state.register_buffer(name, value)
    with tempfile.TemporaryDirectory() as folder:
        path = str(Path(folder) / 'coarse.part')
        torch.jit.trace(state, torch.ones(1)).save(path)
        cfg['integration']['implicit-scheme'] = 9
        create(cfg).initialize_from_restart(path)

if __name__ == '__main__':
    coarse_restart()
    failures = []
    for work, fixer in (('cell', True), ('cell', False), ('face', False)):
        for nz in (120, 140, 150):
            out = tall_run(nz, nz * 11.3 / 45, 9, work, 300, fixer=fixer)
            if not (out['finite'] and out['redos'] == 0 and out['steps'] == 300 and abs(out['top_rho']) < 0.01
                    and abs(out['top_T'] / tall.T0 - 1) < 0.01 and out['wmax'] < 0.1
                    and abs(out['mass']) < 1e-12 and out['clamp'] == 0
                    and (work == 'cell' and not fixer or abs(out['epe_drift']) < tall.EPE_TOL)):
                failures.append(f'tall column {nz}: {out}')
        out = tall_run(45, 40, 9, work, 40, 1500, fixer=fixer)
        if work == 'face':
            # Rollback coverage only; a retry does not establish stability.
            if out['redo_status'] > 0:
                if not out['rolled_back']:
                    failures.append(f'coarse face retry did not restore its input: {out}')
            elif not (out['finite'] and out['steps'] == 40 and out['wmax'] < 0.1
                      and abs(out['mass']) < 1.e-12):
                failures.append(f'coarse face run outside rollback control: {out}')
        elif not (out['finite'] and out['redos'] == 0 and out['steps'] == 40
                  and out['wmax'] < 0.1 and abs(out['mass']) < 1.e-12):
            failures.append(f'coarse cell column: {out}')
        out = tall_run(160, 40, 9, work, 40, 1500, fixer=fixer)
        if not (out['finite'] and out['redos'] == 0 and out['steps'] == 40 and out['wmax'] < 0.1
                and abs(out['mass']) < 1e-12
                and (work == 'cell' and not fixer or abs(out['epe_drift']) < tall.EPE_TOL)):
            failures.append(f'coarse column 160: {out}')
    for scheme in (0, 1, 9):
        for (work, fixer) in (('cell', False), ('cell', True), ('face', False)):
            out = solid_run(scheme, work, fixer)
            if not (out['finite'] and abs(out['mass']) < 1e-12):
                failures.append(f'solid mass: {out}')
    for placement in ('middle', 'bottom'):
        for scheme in (1, 9):
            for work in ('cell', 'face'):
                out = solid_run(scheme, work, False, placement)
                if not (out['finite'] and abs(out['mass']) < 1e-12):
                    failures.append(f'segmented solid mass: {out}')
    for scheme in (1, 9):
        out = solid_run(scheme, 'face', False, strided=True)
        if not out['finite'] or out['mass'] >= 1.e-12:
            failures.append(f'transposed solid mask: {out}')
    for scheme in (1, 9):
        for geometry in ('gnomonic-equiangle', 'spherical-polar'):
            for work in ('cell', 'face'):
                out = curved_energy(scheme, geometry, work)
                if not out['finite'] or not out['relative'] <= 1.e-12:
                    failures.append(f'curved implicit energy: {out}')
    for scheme in (1, 9):
        for work in ('cell', 'face'):
            defect = clamp_energy(scheme, work)
            if abs(defect) > 1e-10:
                failures.append(f'clamped energy scheme {scheme}, {work}: {defect}')
    for failure in failures:
        print('FAIL', failure)
    sys.exit(bool(failures))
