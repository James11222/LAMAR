// Import Packages
#include <stdlib.h>
#include <time.h>
#include <stdio.h>
#include <math.h>
#include <unistd.h>
#include <assert.h>
#include <iostream>
#include <fstream>

/* - - - - - - - - - - - - - - 
            GLOBALS
- - - - - - - - - - - - - - - - */ 
 
float G = 4.49850215e-15; // pc^3 / (solMass yr^2)
int Nparticles;
float dt;
float tmax;
int skip_Nsteps;

// ####################################################################
//                         PHYSICS FUNCTIONS
// ####################################################################

/** 
    * @brief Function to update the position of particle i
    * 
    * @param position: position of the particles
    * @param acceleration: acceleration of the particles
    * @param potential: potential energy of the particles
    * @param Nparticles: number of particles in the simulation
    * @param G: gravitational constant
    * @param dt: time step size
    *
    * @return void
*/ 

__global__ void update_position(float4 *position, float3 *velocity, float3 *acceleration, int Nparticles, float G, float dt, bool final_drift)
{ 
    // Get the index of the thread being used for this particle
    int i = threadIdx.x + (blockIdx.x *blockDim.x);

    // Check if the thread i is greater than the number of particles in the simulation
    if(i >= Nparticles)
        return;

    // Get the position, velocity, and acceleration of the particle
    // We define pointers to the position, velocity, and acceleration of the particle
    float3 *vi = &velocity[i];     // velocity of the particle
    float4 *pi = &position[i];     // position of the particle, 4th element is mass
    float3 *ai = &acceleration[i]; // acceleration of the particle

    // Time Integration

    // - - - - - - - - - - - - - - - - - 
    //   Symplectic Euler Integration
    // - - - - - - - - - - - - - - - - - 

    // // Update the velocity of the particle
    // vi->x = vi->x + ai->x*dt;
    // vi->y = vi->y + ai->y*dt;
    // vi->z = vi->z + ai->z*dt;

    // // Update the position of the particle using the updated velocity
    // pi->x = pi->x + vi->x*dt;
    // pi->y = pi->y + vi->y*dt;
    // pi->z = pi->z + vi->z*dt;

    // - - - - - - - - - - - - - - - - - 
    //        Verlet Integration
    // - - - - - - - - - - - - - - - - - 
    
    // drift
    if (final_drift == false)
    {
        // drift
        vi->x = vi->x + 0.5 * ai->x * dt; //v^n+1/2
        vi->y = vi->y + 0.5 * ai->y * dt; //v^n+1/2
        vi->z = vi->z + 0.5 * ai->z * dt; //v^n+1/2

        //kick
        pi->x = pi->x + vi->x * dt; //x^n+1
        pi->y = pi->y + vi->y * dt; //x^n+1
        pi->z = pi->z + vi->z * dt; //x^n+1
    }
    else
    {
        //drift
        vi->x = vi->x + 0.5 * ai->x * dt; //v^n+1
        vi->y = vi->y + 0.5 * ai->y * dt; //v^n+1
        vi->z = vi->z + 0.5 * ai->z * dt; //v^n+1
    }

    return;
}


/**
    * @brief a function to calculate the total force on the the ith particle
    * 
    * @param position 
    * @param acceleration 
    * @param potential 
    * @param Nparticles 
    * @param G 
    * @param dt 
    * 
    * @return void 
 */
__global__ void calcForce(float4 *position, float3 *acceleration, float *potential, int Nparticles, float G, float dt)
{
    // Get the index of the thread being used for this particle
    int i = threadIdx.x + (blockIdx.x *blockDim.x);

    // Check if the thread i is greater than the number of particles in the simulation
    if(i >= Nparticles)
        return;

    // Get the position, acceleration, and potential of the particle
    // We define pointers to the position, acceleration, and potential of the particle
    float4 *pi  = &position[i];     // position of the particle, 4th element is mass
    float3 *ai  = &acceleration[i]; // acceleration of the particle
    float *poti = &potential[i];    // potential energy of the particle

    // Initialize necessary local variables
    int j; float4 pj;
    float x, y, z;
    float dx, dy, dz, a;
    float r, r_squared, r_cubed;
    float ax, ay, az;
    float pot;
    float epsilon;


    // Get the position of the particle i
    x = pi->x;
    y = pi->y;
    z = pi->z;

    // Initialize the acceleration components to zero and potential to zero
    ax  = 0.0;
    ay  = 0.0;
    az  = 0.0;
    pot = 0.0; 
    epsilon = 100.0; //softening parameter

    // loop over all the other particles
    for(j = 0; j < Nparticles; j++)
    {
        // Get the position of the particle j
        pj = position[j];
        
        // skip the particle if it is the same as the particle we are calculating the force on
        if(j == i){
            continue;
        }

        // compute the distance between the two particles
        dx = pj.x - x;
        dy = pj.y - y;
        dz = pj.z - z;

        // compute the squared distance between the two particles
        r_squared = (dx*dx)+(dy*dy)+(dz*dz)+(epsilon*epsilon);
        r = sqrtf(r_squared);
        r_cubed = r * r_squared;

        // scalar acceleration computed using Newton's law of gravitation
        a = ((G*pj.w)/(r_cubed));

        // update the acceleration components using the computed acceleration
        ax += a*dx;
        ay += a*dy;
        az += a*dz;

        // update the potential energy of the particle
        pot -= (G*pj.w)/(r);
    }

    // update the acceleration and the potential energy of the ith particle 
    ai->x = ax;
    ai->y = ay;
    ai->z = az;
    *poti = pot;

    return;
}

// ####################################################################
//                         FILE I/O FUNCTIONS
// ####################################################################

/**
    * @brief a function to write the position, velocity, and potential energy of the particles to a file
    * 
    * @param pos 
    * @param vel 
    * @param pot 
    * 
    * @return int 
 */
int toFile(float4 *pos, float3 *vel, float *pot)
{
    // initialize the necessary variables
	int i;
	float4 p;
	float3 pv;
	float pp;

    // open the files
	FILE *fx = fopen("outputs/x.dat","a");
	FILE *fy = fopen("outputs/y.dat","a");
	FILE *fz = fopen("outputs/z.dat","a");
	FILE *fm = fopen("outputs/m.dat","a");

	FILE *fvx = fopen("outputs/vx.dat","a");
	FILE *fvy = fopen("outputs/vy.dat","a");
	FILE *fvz = fopen("outputs/vz.dat","a");

	FILE *fp = fopen("outputs/pot.dat","a");

    // loop over all the particles
	for(i = 0; i < Nparticles; i++)
	{
        // get the position, velocity, and potential energy of the particle
		p  = pos[i];
		pv = vel[i];
		pp = pot[i];

		if(p.w > 0)
		{
			fprintf(fx, "%e\t", p.x);
			fprintf(fy, "%e\t", p.y);
			fprintf(fz, "%e\t", p.z);
			fprintf(fm, "%e\t", p.w);

			fprintf(fvx, "%e\t", pv.x);
			fprintf(fvy, "%e\t", pv.y);
			fprintf(fvz, "%e\t", pv.z);

			fprintf(fp, "%e\t", pp);
		}
	}
    // print position and masses
	fprintf(fx, "\n"); fprintf(fy, "\n"); fprintf(fz, "\n"); fprintf(fm, "\n");

    // print velocities
	fprintf(fvx, "\n"); fprintf(fvy, "\n"); fprintf(fvz, "\n");

    // print potential energy
	fprintf(fp, "\n");

    // close the files
	fclose(fx);fclose(fy);fclose(fz);fclose(fm);
	fclose(fvx);fclose(fvy);fclose(fvz);
	fclose(fp);

	return 0;
}


/** 
    * @brief a function to clear the files
    * @return void
*/
void clearFile()
{
    // open the files
	FILE *fx  = fopen("outputs/x.dat","w");
	FILE *fy  = fopen("outputs/y.dat","w");
	FILE *fz  = fopen("outputs/z.dat","w");
	FILE *fvx = fopen("outputs/vx.dat","w");
	FILE *fvy = fopen("outputs/vy.dat","w");
	FILE *fvz = fopen("outputs/vz.dat","w");
	FILE *fm  = fopen("outputs/m.dat","w");
	FILE *fp  = fopen("outputs/pot.dat","w");

    // close the files
	fclose(fx);fclose(fy);fclose(fz);fclose(fm);
	fclose(fvx);fclose(fvy);fclose(fvz);
	fclose(fp);
}

int main(int argc, char  ** argv)
{ 
    float t;         // time
    int n;           // number of time steps taken
    int i;           // particle index

    // open the file
    std::ifstream inFile(argv[1]);

    inFile >> Nparticles >> tmax >> dt >> skip_Nsteps;
    // read the number of particles, time step, and number of steps
    printf("N particles: %d\n", Nparticles);
	printf("t max: %5.2e\n", tmax);
	printf("dt: %5.2e\n", dt);
	printf("N steps per write: %d\n", skip_Nsteps);

    // printf("Init\n");
	clearFile();
	cudaDeviceReset();

    // define variables for position, velocity, and potential energy of the particles on the GPU and CPU
    float4 *position_GPU, *position_CPU;
    float3 *acceleration_GPU;
    float3 *velocity_GPU, *velocity_CPU;
    float  *potential_GPU, *potential_CPU;

    // allocate memory on the CPU
	position_CPU  = (float4*)malloc(sizeof(float4)*Nparticles);
	velocity_CPU  = (float3*)malloc(sizeof(float3)*Nparticles);
	potential_CPU = (float*)malloc(sizeof(float)*Nparticles);

    // allocate memory on the GPU
    cudaError_t rc = cudaMalloc((void**)&position_GPU, Nparticles*sizeof(float4));
    rc = cudaMalloc((void**)&velocity_GPU, Nparticles*sizeof(float3));
    rc = cudaMalloc((void**)&acceleration_GPU, Nparticles*sizeof(float3));
    rc = cudaMalloc((void**)&potential_GPU, Nparticles*sizeof(float));

    // read the initial positions, velocities, and masses of the particles
    float x, y, z, vx, vy, vz, m;
	i = 0;
	while (inFile >> x >> y >> z >> vx >> vy >> vz >> m) {
		position_CPU[i].x = x; position_CPU[i].y = y;position_CPU[i].z = z;position_CPU[i].w = m;
		velocity_CPU[i].x = vx; velocity_CPU[i].y = vy ;velocity_CPU[i].z = vz;
		potential_CPU[i] = -1;
		i++;
	}
    i = 0;

    // transfer the initial positions, velocities, and masses of the particles to the GPU
	cudaMemcpy(position_GPU, position_CPU,  Nparticles*sizeof(float4), cudaMemcpyHostToDevice);
	cudaMemcpy(velocity_GPU, velocity_CPU,  Nparticles*sizeof(float3), cudaMemcpyHostToDevice);
	cudaMemcpy(potential_GPU,potential_CPU, Nparticles*sizeof(float),  cudaMemcpyHostToDevice);

    // set up the GPU architecture 
    int threads = 1024;
    int blocks = (int)(ceil(Nparticles/threads))+1;
    dim3 grid(blocks,1);
    dim3 block(threads,1,1);

    // calculate the forces on the particles for the first time step so we can rewrite what the 
    // proper initial potential is for all the particles. 
    calcForce<<<grid, block>>>(position_GPU, acceleration_GPU, potential_GPU, Nparticles, G, dt);
    cudaMemcpy(potential_CPU, potential_GPU, Nparticles * sizeof(float),  cudaMemcpyDeviceToHost);
    toFile(position_CPU, velocity_CPU, potential_CPU);

    t = 0.0; n = 0; // initialize the time and number of time steps taken
    // loop over all the time steps
    while(t <= tmax)
    {
        // every skip_Nsteps we write the positions, velocities, and potential energies to a file
        if ((n%skip_Nsteps) == 0 && n != 0)
        {
            // printf(" %d", (int)n/skip_Nsteps);
            printf("t = %5.4e years\n", t);
            // transfer a copy of GPU data to CPU
            cudaMemcpy(potential_CPU, potential_GPU, Nparticles * sizeof(float),  cudaMemcpyDeviceToHost);
            cudaMemcpy(position_CPU, position_GPU, Nparticles * sizeof(float4), cudaMemcpyDeviceToHost);
            cudaMemcpy(velocity_CPU, velocity_GPU, Nparticles * sizeof(float3), cudaMemcpyDeviceToHost);

            // write the data to a file
            toFile(position_CPU, velocity_CPU, potential_CPU);
            // printf("\n");
        }

        // calculate the forces on the particles
        calcForce<<<grid, block>>>(position_GPU, acceleration_GPU, potential_GPU, Nparticles, G, dt);
        cudaDeviceSynchronize();

        // take a half step in velocity and then update the positions of the particles
        update_position<<<grid, block>>>(position_GPU, velocity_GPU, acceleration_GPU, Nparticles, G, dt, false);
        cudaDeviceSynchronize();

        // recalculate the forces on the particles with new positions
        calcForce<<<grid, block>>>(position_GPU, acceleration_GPU, potential_GPU, Nparticles, G, dt);
        cudaDeviceSynchronize();

        // update the velocities fully using the newly calculated forces
        update_position<<<grid, block>>>(position_GPU, velocity_GPU, acceleration_GPU, Nparticles, G, dt, true);
        cudaDeviceSynchronize();

        t += dt; n += 1; // update the time and number of time steps taken
    }
    return 0;
}

