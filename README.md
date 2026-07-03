# GPU Fluid Simulation in JavaScript & WebGPU

An interactive Eulerian fluid simulation implemented in JavaScript with WebGPU compute shaders. The solver runs on the GPU and supports real-time density and velocity splats, adjustable grid resolutions, vorticity confinement, and configurable diffusion/decay.

The goal of this project is to learn [WebGPU](https://webgpu.org) by building a highly parallel physics simulation that runs directly in the browser.

[Access the simulation here](https://freddycoppa.github.io/webgpu-fluid-sim/)

![Fluid simulation screenshot](favicon.png)

## Features

1. Separate density and velocity grid scales for balancing visual quality and simulation performance
2. Interactive Gaussian splats for injecting density and velocity with the mouse
3. Random splats triggered by space key or button input
4. Adjustable vorticity confinement to preserve small-scale rotational detail
5. Adjustable density and velocity decay
6. Adjustable shading options, including contour lines and sharpening, for enhanced fluid visualization

## Theory

This simulation solves the incompressible inviscid Navier-Stokes equations (a.k.a. Euler equations) on a 2-dimensional fluid velocity field. For a velocity field $\vec u$, the Euler equations are as follows:

$$
\begin{gather}
\frac{\partial \vec u}{\partial t} + \left(\vec u \cdot \nabla\right) \vec u = - \frac{1}{\rho} \nabla p + \vec f \\
\nabla \cdot \vec u = 0
\end{gather}
$$

Equation $(1)$ says the acceleration of the fluid depends on:
1. **Advection** $\left(\vec u \cdot \nabla\right) \vec u$: Each fluid parcel is carried by the fluid's velocity at the parcel's location.
2. **Pressure Gradient** $- \frac{1}{\rho} \nabla p$: The fluid flows toward the negative pressure gradient, or the direction of steepest decrease in pressure. The constant $\rho$ is the density of the fluid.
3. **Total External Force** $\vec f$.

Equation $(2)$ states that the velocity field is divergence free, i.e., the fluid is incompressible. At any point in the fluid, the total flow out is equal to the total flow in.

## Mechanism

The fluid is represented as a 2-dimensional discrete velocity field. Every frame, the simulation iteratively solves the Euler equations on this discrete velocity grid. To visualize the fluid, an additional 2D discrete density field is implemented to simulate a colorful dye being mixed into the fluid and carried by its currents, eddies, and vorticies.

The main way to interact with the fluid is by splatting. Dragging the mouse pointer over the fluid triggers a gaussian splat of a random dye color onto the dye density field, as well as a gaussian splat of the mouse velocity onto the velocity field, at the location of the mouse pointer. The simulation updates in real time to solve these new density & velocity fields. This gives the effect of mixing more dye into the fluid.

## Data Structures

This simulation mainly uses 2-dimensional storage textures as the underlying data structure for the dye density and velocity fields,
to take advantage of GPU hardware texture indexing. Unfortunately, as far as WebGPU is concerned,
hardware sampling of 32-bit floating point textures is not guaranteed to be supported on all GPUs, so software
sampling is used.

In a texture, each texel corresponds to one discrete cell of the field the texture represents. The dye density texture stores 16-bit floating point RGB values to provide a wide range of dye colors while maintaining precision, whereas the velocity textures store the horizontal and vertical components of velocity as 32-bit floats.

Since dye density and velocity are stored in separate textures, they can be computed at different scales.
The dye density field is used to visualize the fluid on the screen, meaning its texture dimensions would usually match the
screen resolution. On the other hand, the velocity field is used to simulate the physics of the fluid, so its dimensions
can be scaled down to improve performance. The dimensions of both the density and velocity fields can be adjusted from
the control panel via the "Display resolution" and "Simulation scale" dropdowns respectively.

The velocity field is implemented as a staggered grid, where the horizontal components of the velocities pass through the
vertical faces of each cell, and the vertical components of the velocities pass through the horizontal faces of each cell.
This is done by separately storing a `u` texture for the horizontal velocity components, and a `v` texture for the vertical
velocity components (both are scalar 32-bit float textures). For a velocity field of dimensions `width * height`:

1. The texture `u` would have dimensions `(width + 1) * height`, corresponding to every vertical cell wall in the velocity field, and
2. The texture `v` would have dimensions `width * (height + 1)`, corresponding to every horizontal cell wall in the velocity field.

A staggered velocity grid provides the following advantages:

1. Divergence is naturally computed at the center of a cell via the net velocity flow through all of its faces.
2. Curl is naturally computed at the corner of a cell via the net velocity flow around the point (through every face adjacent to that corner).
3. When performing pressure projection, the pressure gradient naturally aligns with cell faces, making it easy to subtract from each velocity component.

### Ping-Pong Textures

Since this an iterative algorithm, it involves updating the same fields every timestep. To avoid race conditions that arise when reading from and writing to the same texture, simulation employs double-buffering. Every field corresponds to two physical textures on the GPU — `in` and `out`. Each frame, a GPU kernel reads from `in`, performs the relevant computations, and writes the result to `out`. At the end of the frame, the buffers are swapped, and the whole process repeats.

## Algorithms

The simulation follows the advect, confine, project, splat loop. Every frame, the fluid is solved via the following steps:

### Density & Velocity Advection
Advection basically entails moving every value in a field along its local velocity. For example, let's consider a point $(x, y)$ in a field $F$ with local value $F(x, y) = \alpha$ and velocity $(v_x, v_y)$. After a timestep $dt$, we would expect the value $\alpha$ to end up at the point $(x + v_x \cdot dt, y + v_y \cdot dt)$. In other words, $F(x + v_x \cdot dt, y + v_y \cdot dt) = \alpha$. For a discrete field represented as a 2D texture `F`, the corresponding update rule would be
```
F[i + vi * dt, j + vj * dt] <- F[i, j]
```
for index `[i, j]`, velocity `(vi, vj)`, and timestep `dt`.

However, since we're running advection on the GPU, we need the algorithm to be local — a kernel processing input cell `[i, j]` should only write to output cell `[i, j]`. Therefore, instead of tracing the velocity at `[i, j]` *forward* by timestep `dt`, we trace it *backward*, and see what value lives there. In other words, our update rule becomes
```
F[i, j] <- F[i - vi * dt, j - vj * dt]
```
A good way to think about this is — at a given point, instead of figuring out where the local velocity *pushes* its own value to, we're trying to probe what value this velocity *pulls* into its own cell.
This is called **Semi-Lagrangian advection**.

Of course, an important consideration is that the index `[i - vi * dt, j - vj * dt]` may not be integral. To access a value from a texture at a fractional index, we employ **bilinear sampling**.

The start of one iteration of the simulation loop involves advecting the density and velocity fields. The density field is advected along the velocity field, and the velocity field is advected along itself.

### Vorticity Confinement

Semi-Lagrangian advection tends to blur out the fine details of the velocity field, including its vortices. Vorticity confinement aims to add these vortices back.

The first step to vorticity confinement is computing the curl of the velocity field after advection. As discussed earlier, on a staggered velocity grid, the curl lives at cell corners.
Every cell corner has a `u` face on top, a `u` face on the bottom, a `v` face to its left, and a `v` face to its right (remember `u` faces are vertical faces representing horizontal velocity components, and `v` faces are horizontal faces representing vertical velocity components). Let's call these faces `u_top`, `u_bottom`, `v_left` & `v_right` respectively.

The curl of the velocity field is mathematically defined as

$$
\nabla \times \left(u, v\right) =\frac{\partial v}{\partial x} - \frac{\partial u}{\partial y}
$$

This translates elegantly as
```
curl(cell_corner) = v_right - v_left - u_top + u_bottom
```
This formula makes intuitive sense if you visualize how the velocities around a cell corner make it spin. Clockwise curl is negative, and counterclockwise curl is positive.

Once we've computed the curl, we compute the gradient of its absolute value $\nabla \left| \nabla \times \vec u \right|$. This gives us the direction towards the steepest increase in vorticity magnitude from every location. To confine vorticity at a point, we want to reinforce the strongest vortex around it, so we add a force to the velocity field perpendicular to this gradient.

### Projection

Advection (and splatting) usually introduce divergence into the velocity field. To remove divergence from the velocity field, we "project" it onto a divergence-free subspace of vector fields (similar to least squares, wherein we find the closest possible approximation to an unsolvable system). This comes from the Helmholtz Decomposition Theorem, which states that any vector field $F$ can be written as the sum of a solenoidal (divergence-free) field and an irrotational (curl-free) field:

$$
F = \nabla \times A + \nabla \phi
$$

Here, $A$ is a vector potential and $\phi$ is a scalar potential. $\nabla \times A$ is divergence-free because the divergence of curl is zero: $\nabla \cdot \nabla \times A = 0$, and $\nabla \phi$ is curl-free because the curl of gradient is zero: $\nabla \times \nabla \phi = 0$.

#### Pressure Poisson Equation

Let our advected velocity field *with* divergence be $u^*$. We want to solve for a pressure gradient $\nabla p$, such that after subtracting, the divergence of the remaining field is zero:

$$\nabla \cdot (u^* - \nabla p) = 0$$

If we wrangle this equation a little, we arrive at the pressure Poisson equation:

$$
\begin{align*}
    &\nabla \cdot (u^* - \nabla p) = 0 \\
    \implies& \nabla \cdot u^* - \nabla^2 p = 0 \\
    \implies& \nabla^2 p = \nabla \cdot u^*
\end{align*}
$$

In discrete terms, we want to solve for a pressure potential $p$ such that for all $i,j$:

$$
\begin{gather*}
p_{i+1,j} + p_{i-1,j} + p_{i,j+1} + p_{i,j-1} - 4p_{i,j} = d_{i,j} \\
\implies p_{i,j} = \frac14\left(p_{i+1,j} + p_{i-1,j} + p_{i,j+1} + p_{i,j-1} - d_{i,j}\right)
\end{gather*}
$$

Where $d_{i,j}$ is the divergence of the velocity field at cell $(i,j)$. This rearrangement hints at an iterative method to solve for the whole pressure potential. Every iteration, the new pressure value is computed based on the surrounding old pressure values and the divergence.

#### Divergence

Since our discrete velocity field lies on the faces of the cells of the simulation, calculating divergence becomes trivial. For a cell whose left and right faces are $u_\text{left}$ and $u_\text{right}$, and whose top and bottom faces are $v_\text{top}$ and $v_\text{bottom}$, its divergence is computed as

$$
\text{div}(\text{cell}) = \nabla \cdot (u, v) = \frac{\partial u}{\partial x} + \frac{\partial v}{\partial y} = u_\text{right} - u_\text{left} + v_\text{top} - v_\text{bottom}
$$

#### Red-Black Gauss-Seidel

The regular method for solving the pressure field is Jacobi iteration, where in we use a double buffer to compute the next pressure field given its current value. This converges fairly slowly, so this simulation employs Gauss-Seidel projection to speed up convergence.

Gauss-Seidel is exactly like Jacobi, except that it only uses a single buffer, and updates each cell using the newest values available. So instead of reading from one pressure buffer and writing the updated values to another, it reads from and writes to the same buffer, performing subsequent updates using newly updated values. This dramatically speeds up convergence.

The only problem here is that naive Gauss-Seidel can't be implemented on a GPU, because multiple kernels reading from and writing to the same texture simultaneously would cause a race condition. The solution here is to use red-black Gauss-Seidel.

Red-Black Gauss-Seidel marks cells with even parity as red, and cells with odd parity as black. This yields a checkerboard grid of red and black cells. Observe that a red cell only accesses its black neighbors during a pressure update, and vice-versa. This means we can perform Gauss-Seidel only on red cells, and then only on black cells, without worrying about race conditions. This is essentially red-black Gauss-Seidel.

## Footnotes

1. This simulation doesn't exactly perform full-blown pressure projection. Instead, it does a sort of iterative divergence relaxation, which removes divergence from the velocity field without explicitly solving for the pressure potential. I learned this method from [matthias-research](https://matthias-research.github.io/pages/tenMinutePhysics/17-fluidSim.pdf).

## Other Topics

1. Coordinate system & sampling the fields
2. Boundary conditions

## Future Work
1. Pretty shading
2. Obstacles, non-rectangular fluid simulation
3. Additional visualization modes for velocity, pressure, divergence & curl
4. Benchmarking & performance improvements
5. Exportable screenshots/recordings
