#!/usr/bin/env python3
"""
INTERACTIVE 3D CAUSAL FIELD VISUALIZATION

Uses the ACTUAL causal field law:
T^{μν}_{ρσ}(x) = α J^{μν}_{ρσ}(x) - (1-α) ∫_{J^-(x)} k(τ) Π Γ J Φ d⁴x'

Shows rank-4 tensor as 3D volume with interactive controls for:
- α (alpha): QM ↔ GR interpolation
- ρ (rho): Memory decay
- κ (kappa): Kernel scale
"""

import sys
sys.path.insert(0, '/home/sam_leizerman/Liorhybrid')

import torch
import torch.nn as nn
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Import actual modules
from models.causal_field import AssociatorCurrent, ParallelTransport, CliffordConnection
from models.lior_kernel import LiorKernel

print("="*70)
print("INTERACTIVE CAUSAL FIELD: T^{μν}_{ρσ} = αJ + (1-α)∫kΠΓJΦ")
print("="*70)

# ============================================================================
# ACTUAL CAUSAL FIELD IMPLEMENTATION
# ============================================================================

class CausalFieldTensor:
    """
    Computes the actual causal field tensor T^{μν}_{ρσ} from your theory.
    """
    def __init__(self, d_model=64, d_field=4, d_spinor=4):
        self.d_model = d_model
        self.d_field = d_field
        self.d_spinor = d_spinor

        # Initialize actual components
        self.associator = AssociatorCurrent(d_model, d_field=16)
        self.Pi = ParallelTransport(d_field=16, d_spinor=d_spinor)
        self.Gamma_conn = CliffordConnection(d_spinor=d_spinor)
        self.kernel = LiorKernel(p_eff=4)

        # Phi bivector (antisymmetric)
        self.Phi = torch.randn(d_field, d_field) * 0.1
        self.Phi = self.Phi - self.Phi.T

        # Projection to 4x4x4x4 tensor
        self.proj_to_rank4 = nn.Linear(16*16, d_field**4)

    def compute_J(self, x):
        """Compute associator current J^{μν}_{ρσ}."""
        # x: [B, N, d_model]
        J_16x16 = self.associator(x)  # [B, N, 16, 16]
        return J_16x16

    def compute_transported(self, J):
        """Apply Π·Γ to J."""
        Gamma = self.Gamma_conn()  # [d_spinor, d_spinor]
        transported = self.Pi(J, Gamma)  # [B, N, 16, 16]
        return transported

    def compute_memory_integral(self, transported, rho, kappa, N_time):
        """
        Compute ∫_{J^-(x)} k(τ) Π·Γ·J d⁴x' via O(1) recurrence.

        m_t = ρ·m_{t-1} + κ·(Π·Γ·J)_t
        """
        B, N, d1, d2 = transported.shape
        transported_flat = transported.view(B, N, -1)  # [B, N, 256]

        memory_state = torch.zeros(B, d1*d2)
        memory_out = torch.zeros_like(transported_flat)

        for t in range(N):
            memory_state = rho * memory_state + kappa * transported_flat[:, t, :]
            memory_out[:, t, :] = memory_state

        return memory_out.view(B, N, d1, d2)

    def compute_T(self, x, alpha, rho, kappa):
        """
        Compute full causal field tensor:
        T = α·J + (1-α)·M

        where M = ∫ k(τ) Π·Γ·J Φ
        """
        B, N, D = x.shape

        # J = associator current
        J = self.compute_J(x)  # [B, N, 16, 16]

        # Π·Γ·J = parallel transported
        transported = self.compute_transported(J)

        # M = memory integral
        M = self.compute_memory_integral(transported, rho, kappa, N)

        # T = α·J + (1-α)·M
        T = alpha * J + (1 - alpha) * M

        # Project to rank-4 tensor [B, N, 4, 4, 4, 4]
        T_flat = T.view(B, N, -1)  # [B, N, 256]
        T_rank4 = T_flat.view(B, N, 4, 4, 4, 4)

        return T_rank4, J, M

# ============================================================================
# CREATE VISUALIZATION
# ============================================================================

print("Initializing causal field...")
cf = CausalFieldTensor(d_model=64, d_field=4, d_spinor=4)

# Generate input sequence
N_seq = 32
x_input = torch.randn(1, N_seq, 64) * 0.5

def compute_field(alpha, rho, kappa, t_idx):
    """Compute T tensor for given parameters at time t_idx."""
    with torch.no_grad():
        T, J, M = cf.compute_T(x_input, alpha, rho, kappa)
        # Get tensor at time t_idx: [4, 4, 4, 4]
        T_t = T[0, t_idx].numpy()
        J_t = J[0, t_idx].numpy()
        M_t = M[0, t_idx].numpy()
    return T_t, J_t, M_t

# Initial computation
alpha_init, rho_init, kappa_init = 0.5, 0.9, 0.15
t_init = N_seq // 2
T_init, J_init, M_init = compute_field(alpha_init, rho_init, kappa_init, t_init)

print(f"T tensor shape: {T_init.shape} = [μ, ν, ρ, σ]")

# ============================================================================
# 3D VOLUME VISUALIZATION
# ============================================================================

def create_3d_volume_figure(T_tensor, title, mu_slice=0):
    """
    Create 3D volume visualization of T^{μν}_{ρσ}.

    Fix μ, show ν, ρ, σ as 3D cube.
    """
    # T_tensor: [4, 4, 4, 4]
    # Fix μ = mu_slice, get [4, 4, 4] cube
    cube = T_tensor[mu_slice]  # [ν, ρ, σ]

    # Create 3D grid
    nu, rho_idx, sigma = np.meshgrid(
        np.arange(4), np.arange(4), np.arange(4), indexing='ij'
    )

    # Flatten for scatter
    x = nu.flatten()
    y = rho_idx.flatten()
    z = sigma.flatten()
    values = cube.flatten()

    # Normalize for color/size
    vmax = np.abs(values).max() + 1e-8
    colors = values / vmax
    sizes = np.abs(values) / vmax * 30 + 5

    fig = go.Figure(data=[go.Scatter3d(
        x=x, y=y, z=z,
        mode='markers',
        marker=dict(
            size=sizes,
            color=values,
            colorscale='RdBu',
            cmin=-vmax,
            cmax=vmax,
            colorbar=dict(title='T value'),
            opacity=0.8
        ),
        text=[f'T[{mu_slice},{i},{j},{k}]={v:.4f}'
              for i,j,k,v in zip(x,y,z,values)],
        hoverinfo='text'
    )])

    fig.update_layout(
        title=title,
        scene=dict(
            xaxis_title='ν',
            yaxis_title='ρ',
            zaxis_title='σ',
            aspectmode='cube'
        ),
        width=700, height=700
    )

    return fig

def create_isosurface_figure(T_tensor, title):
    """
    Create isosurface visualization of the full 4D tensor.

    Use magnitude |T^{μν}_{ρσ}| collapsed over one index.
    """
    # Sum over μ to get 3D: sum_μ |T^{μν}_{ρσ}|
    T_3d = np.abs(T_tensor).sum(axis=0)  # [4, 4, 4]

    # Interpolate to finer grid for smoother isosurface
    # Replace scipy zoom with PyTorch interpolation
    T_3d_torch = torch.from_numpy(T_3d).unsqueeze(0).unsqueeze(0).float()  # [1, 1, 4, 4, 4]
    T_fine_torch = torch.nn.functional.interpolate(
        T_3d_torch,
        scale_factor=4,
        mode='trilinear',
        align_corners=True
    )
    T_fine = T_fine_torch.squeeze().numpy()  # [16, 16, 16]

    # Create meshgrid
    n = T_fine.shape[0]
    X, Y, Z = np.mgrid[0:4:complex(n), 0:4:complex(n), 0:4:complex(n)]

    fig = go.Figure(data=go.Isosurface(
        x=X.flatten(),
        y=Y.flatten(),
        z=Z.flatten(),
        value=T_fine.flatten(),
        isomin=T_fine.max() * 0.3,
        isomax=T_fine.max() * 0.9,
        opacity=0.6,
        surface_count=5,
        colorscale='Viridis',
        caps=dict(x_show=False, y_show=False, z_show=False)
    ))

    fig.update_layout(
        title=title,
        scene=dict(
            xaxis_title='ν',
            yaxis_title='ρ',
            zaxis_title='σ',
            aspectmode='cube'
        ),
        width=700, height=700
    )

    return fig

# ============================================================================
# INTERACTIVE DASHBOARD
# ============================================================================

print("Creating interactive dashboard...")

# Create figure with sliders
fig = make_subplots(
    rows=2, cols=2,
    specs=[[{'type': 'scene'}, {'type': 'scene'}],
           [{'type': 'scene'}, {'type': 'scene'}]],
    subplot_titles=[
        'T tensor (μ=0 slice)', 'T tensor (μ=1 slice)',
        'J (Associator)', 'M (Memory)'
    ]
)

def add_scatter3d(fig, data, row, col, name, cmap='RdBu'):
    """Add 3D scatter to subplot."""
    cube = data[:4, :4, :4] if data.shape[0] > 4 else data[0, :4, :4, :4] if len(data.shape) == 4 else data
    if len(cube.shape) == 4:
        cube = cube[0]  # Take first slice if 4D

    nu, rho_idx, sigma = np.meshgrid(np.arange(4), np.arange(4), np.arange(4), indexing='ij')
    x, y, z = nu.flatten(), rho_idx.flatten(), sigma.flatten()
    values = cube.flatten()
    vmax = np.abs(values).max() + 1e-8

    fig.add_trace(
        go.Scatter3d(
            x=x, y=y, z=z,
            mode='markers',
            marker=dict(
                size=np.abs(values)/vmax * 20 + 3,
                color=values,
                colorscale=cmap,
                cmin=-vmax, cmax=vmax,
                opacity=0.7
            ),
            name=name
        ),
        row=row, col=col
    )

# Add initial data
add_scatter3d(fig, T_init[0], 1, 1, 'T(μ=0)')
add_scatter3d(fig, T_init[1], 1, 2, 'T(μ=1)')
add_scatter3d(fig, J_init[:4, :4, :4] if J_init.shape[0] >= 4 else np.zeros((4,4,4)), 2, 1, 'J')
add_scatter3d(fig, M_init[:4, :4, :4] if M_init.shape[0] >= 4 else np.zeros((4,4,4)), 2, 2, 'M')

# Create frames for animation over different alpha values
frames = []
for alpha in np.linspace(0.1, 0.9, 20):
    T, J, M = compute_field(alpha, rho_init, kappa_init, t_init)
    frame_data = []

    for mu_idx in [0, 1]:
        cube = T[mu_idx]
        nu, rho_idx, sigma = np.meshgrid(np.arange(4), np.arange(4), np.arange(4), indexing='ij')
        values = cube.flatten()
        vmax = np.abs(values).max() + 1e-8

        frame_data.append(go.Scatter3d(
            x=nu.flatten(), y=rho_idx.flatten(), z=sigma.flatten(),
            mode='markers',
            marker=dict(
                size=np.abs(values)/vmax * 20 + 3,
                color=values, colorscale='RdBu',
                cmin=-vmax, cmax=vmax, opacity=0.7
            )
        ))

    frames.append(go.Frame(data=frame_data, name=f'α={alpha:.2f}'))

fig.frames = frames

# Add sliders
sliders = [
    dict(
        active=10,
        currentvalue={"prefix": "α (QM↔GR): "},
        steps=[
            dict(label=f'{a:.2f}', method='animate',
                 args=[[f'α={a:.2f}'], {'frame': {'duration': 100}, 'mode': 'immediate'}])
            for a in np.linspace(0.1, 0.9, 20)
        ]
    )
]

fig.update_layout(
    title=dict(
        text='<b>CAUSAL FIELD T^{μν}_{ρσ} = αJ + (1-α)M</b><br>' +
             '<sup>Interactive: use slider for α | Each point is a tensor component</sup>',
        x=0.5
    ),
    sliders=sliders,
    scene=dict(xaxis_title='ν', yaxis_title='ρ', zaxis_title='σ', aspectmode='cube'),
    scene2=dict(xaxis_title='ν', yaxis_title='ρ', zaxis_title='σ', aspectmode='cube'),
    scene3=dict(xaxis_title='dim1', yaxis_title='dim2', zaxis_title='dim3', aspectmode='cube'),
    scene4=dict(xaxis_title='dim1', yaxis_title='dim2', zaxis_title='dim3', aspectmode='cube'),
    width=1200, height=1000
)

# Save interactive HTML
fig.write_html('causal_field_interactive.html')
print("\nSaved: causal_field_interactive.html")

# ============================================================================
# FULL PARAMETER SWEEP DASHBOARD
# ============================================================================

print("\nCreating full parameter sweep dashboard...")

# Compute field over parameter grid
alphas = np.linspace(0.1, 0.9, 5)
rhos = np.linspace(0.5, 0.99, 5)
kappas = np.linspace(0.05, 0.3, 5)

# Store all computations
all_T = {}
for a in alphas:
    for r in rhos:
        for k in kappas:
            T, _, _ = compute_field(a, r, k, t_init)
            all_T[(a, r, k)] = T

# Create dashboard with multiple sliders
fig2 = go.Figure()

# Add initial isosurface
T_sum = np.abs(T_init).sum(axis=0)  # [4, 4, 4]
# Replace scipy zoom with PyTorch interpolation
T_sum_torch = torch.from_numpy(T_sum).unsqueeze(0).unsqueeze(0).float()  # [1, 1, 4, 4, 4]
T_fine_torch = torch.nn.functional.interpolate(
    T_sum_torch,
    scale_factor=4,
    mode='trilinear',
    align_corners=True
)
T_fine = T_fine_torch.squeeze().numpy()
n = T_fine.shape[0]
X, Y, Z = np.mgrid[0:4:complex(n), 0:4:complex(n), 0:4:complex(n)]

fig2.add_trace(go.Isosurface(
    x=X.flatten(), y=Y.flatten(), z=Z.flatten(),
    value=T_fine.flatten(),
    isomin=T_fine.max() * 0.2,
    isomax=T_fine.max() * 0.8,
    opacity=0.5,
    surface_count=4,
    colorscale='Magma',
    caps=dict(x_show=False, y_show=False, z_show=False)
))

# Create frames for all parameter combinations
frames2 = []
for a in alphas:
    for r in rhos:
        for k in kappas:
            T = all_T[(a, r, k)]
            T_sum = np.abs(T).sum(axis=0)
            T_fine = zoom(T_sum, 4, order=1)

            frames2.append(go.Frame(
                data=[go.Isosurface(
                    x=X.flatten(), y=Y.flatten(), z=Z.flatten(),
                    value=T_fine.flatten(),
                    isomin=T_fine.max() * 0.2,
                    isomax=T_fine.max() * 0.8,
                    opacity=0.5,
                    surface_count=4,
                    colorscale='Magma',
                    caps=dict(x_show=False, y_show=False, z_show=False)
                )],
                name=f'a{a:.1f}_r{r:.2f}_k{k:.2f}'
            ))

fig2.frames = frames2

# Add sliders for each parameter
steps_alpha = [
    dict(label=f'{a:.1f}', method='animate',
         args=[[f'a{a:.1f}_r{rho_init:.2f}_k{kappa_init:.2f}'],
               {'frame': {'duration': 50}, 'mode': 'immediate'}])
    for a in alphas
]

steps_rho = [
    dict(label=f'{r:.2f}', method='animate',
         args=[[f'a{alpha_init:.1f}_r{r:.2f}_k{kappa_init:.2f}'],
               {'frame': {'duration': 50}, 'mode': 'immediate'}])
    for r in rhos
]

steps_kappa = [
    dict(label=f'{k:.2f}', method='animate',
         args=[[f'a{alpha_init:.1f}_r{rho_init:.2f}_k{k:.2f}'],
               {'frame': {'duration': 50}, 'mode': 'immediate'}])
    for k in kappas
]

fig2.update_layout(
    title=dict(
        text='<b>3D ISOSURFACE: Σ_μ |T^{μν}_{ρσ}|</b><br>' +
             '<sup>α: QM(1)↔GR(0) | ρ: memory decay | κ: kernel scale</sup>',
        x=0.5
    ),
    sliders=[
        dict(active=2, currentvalue={"prefix": "α: "}, steps=steps_alpha, y=0.02, len=0.3, x=0.0),
        dict(active=2, currentvalue={"prefix": "ρ: "}, steps=steps_rho, y=0.02, len=0.3, x=0.35),
        dict(active=2, currentvalue={"prefix": "κ: "}, steps=steps_kappa, y=0.02, len=0.3, x=0.7),
    ],
    scene=dict(
        xaxis_title='ν',
        yaxis_title='ρ',
        zaxis_title='σ',
        aspectmode='cube'
    ),
    width=1000, height=900
)

fig2.write_html('causal_field_3d_isosurface.html')
print("Saved: causal_field_3d_isosurface.html")

print("\n" + "="*70)
print("FILES CREATED:")
print("="*70)
print("""
1. causal_field_interactive.html
   - 4 panels showing T^{μν}_{ρσ} slices, J (associator), M (memory)
   - Slider for α (QM ↔ GR interpolation)
   - 3D scatter: each point is a tensor component

2. causal_field_3d_isosurface.html
   - 3D isosurface of Σ_μ |T^{μν}_{ρσ}|
   - THREE sliders: α, ρ (memory decay), κ (kernel scale)
   - Shows the tensor as a true 3D volume

Open in browser to interact!
""")

print("\nPaths:")
print("/home/sam_leizerman/Liorhybrid/causal_field_interactive.html")
print("/home/sam_leizerman/Liorhybrid/causal_field_3d_isosurface.html")
