#include <stdio.h>
#include<mpi.h>
#include <stdlib.h>
typedef struct{
    float** image_data;
    int m;
    int n;
}
image;


void import_JPEG_file(const char *filename, unsigned char **image_chars, int *image_height, int *image_width, int *num_components);
void export_JPEG_file(const char *filename, unsigned char *image_chars, int image_height, int *image_width, int num_components, int quality);
void convert_float_to_char(unsigned char **image_chars, image *u_bar, int m, int n, int start);
void convert_char_to_float(unsigned char  *pict,image *im,int m, int n, int start);
void iso_diffusion_denoising(image *u, image *u_bar, float kappa, int iters);
void deallocate_image(image *u);
void allocate_image(image *u, int m, int n);


int main(int argc, char *argv[])
{

    int m,n,c,iters;
    int my_m, my_n, my_rank, num_procs, recv_count, my_recv_count, block_size, smallest_block_size;

    float kappa;
    image u, u_bar;
    unsigned char *image_chars, *my_image_chars, *new_image_chars, *my_new_image_chars;
    char *input_jpeg_filename, *output_jpeg_filename;
    my_rank = 0;

    char * kappa_str;
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);
    MPI_Comm_size(MPI_COMM_WORLD, &num_procs);
    int displs[num_procs],recvDispls[num_procs],sendCounts[num_procs], recvCounts[num_procs];
    int i,my_m_rest;
    /*
     *read from command line: kappa,iters,input_jpeg_filename,output_jpeg_filename;
     */
    input_jpeg_filename = argv[1];//riktig char
    output_jpeg_filename = argv[2];//riktig char
    kappa_str = (argv[3]);//maa konvertere til double
    iters = atoi(argv[4]);//maa konvertere til int
    //printf("iters: %d\n",iters);
    kappa = 0.01;//TODO:fix so that kappa can be read from the command line
    kappa = atof(kappa_str);


    if(my_rank==0){
        import_JPEG_file(input_jpeg_filename, &image_chars, &m, &n, &c);
    }

    /////////////////////////////////////////////////////////////////
    //Broadcasts the size from root(=0) to all the other processes.//
    /////////////////////////////////////////////////////////////////
    MPI_Bcast(&m,1,MPI_INT,0,MPI_COMM_WORLD);
    MPI_Bcast(&n, 1, MPI_INT, 0, MPI_COMM_WORLD);
    /*
     *divide the m x n pixels evenly among the MPI processes
     */
    my_n = n;//this is correct
    my_m = (m-2)/num_procs;//without ghost points
    my_m_rest = (m-2)%num_procs;
    smallest_block_size = my_m*my_n;
    if(my_rank<my_m_rest){
        my_m+=1;
    }
    printf("my_m: %d\n", my_m);
    block_size = my_m*n;
    /////////////////////////////////////////////////////////////////////////
    //the last process get a larger my_m if m/num_procs is a decimal number//
    /////////////////////////////////////////////////////////////////////////
//    if(my_rank==num_procs-1){
//        my_m = my_m + (m-2)%num_procs;
//    }
    my_recv_count = my_m*my_n;


    /////////////////////////////////////////////////////
    //this is the picture divided into two processes.
    // n-->
    // ----------------------- m
    // |                     | |
    // |          0          | v
    // -----------------------
    // |                     |
    // |          1          |
    // -----------------------
    ///////////////////////////////////////////////////////
    allocate_image(&u, my_m, my_n);
    allocate_image(&u_bar, my_m, my_n);
    my_image_chars = malloc((block_size+2*n)*(sizeof(int)));


    if(my_rank==0){
        int last_displ=0;
        int current_block_size;
        for(i=0;i<my_m_rest;i++){
            current_block_size = smallest_block_size + n;
            sendCounts[i] = current_block_size + 2*n;
            recvCounts[i] = current_block_size;
            displs[i] = current_block_size*i;
            recvDispls[i] = 0;
            //printf("sendCounts: %d\n", sendCounts[i]);
            printf("displ: %d\n",displs[i]/n);
            last_displ = displs[i];
        }
        printf("rest: %d\n", my_m_rest);
        for(i=my_m_rest;i<num_procs;i++){
            printf("%d\n",i);
            current_block_size = smallest_block_size;
             printf("%d\n", current_block_size);
            sendCounts[i] = current_block_size+2*n;
            recvCounts[i] = current_block_size;
            if(i==0){
                displs[i] = 0;
            }else{
                displs[i] = displs[i-1] + current_block_size;
            }

            recvDispls[i] = 0;
            //printf("sendCounts: %d\n", sendCounts[i]);
            printf("displ: %d\n",  displs[i]/n);
        }
    }

    /*
     *each process asks process 0 for partitiones region
     *of image_chars and copy the values into u
    */
    //MPI_Scatterv(image_chars, sendCounts, displs, MPI_CHAR, my_image_chars, recv_count, MPI_CHAR, 0, MPI_COMM_WORLD);
    //MPI_Scatter(&image_chars, my_m*my_n,MPI_CHAR, &my_image_chars, my_m*my_n, MPI_CHAR, 0,MPI_COMM_WORLD);//assume first that there will be no extra rows
    //MPI_Scatter(image_chars, block_size, MPI_CHAR, my_image_chars, block_size, MPI_CHAR, 0, MPI_COMM_WORLD);
    MPI_Scatterv(image_chars, sendCounts, displs, MPI_CHAR, my_image_chars, block_size+2*n, MPI_CHAR, 0, MPI_COMM_WORLD);



    int start = 0;
    convert_char_to_float(my_image_chars, &u,my_m+2, my_n,start);
    //printf("%f", kappa);
    iso_diffusion_denoising(&u, &u_bar, kappa, iters);

    /*
     *each process sends its resulting content of u_bar to process 0
     *process 0 receives from each process incoming vaules and
     *copy them into the designated region of image_chars
     */

    //convert_float_to_char(&image_chars,&u,my_m, my_n,start);
        int x,y, pict_number,value;
        for(x=0;x<my_m+2;x++){
            for(y=0;y<my_n;y++){
                pict_number = x*n + y;
                value = (int)(u.image_data[x][y]);
                my_image_chars[pict_number] = (unsigned char) value;
            }
        }

        //MPI_Gather(my_image_chars, block_size, MPI_CHAR, image_chars, block_size, MPI_CHAR, 0,MPI_COMM_WORLD);
        //MPI_Gatherv(my_image_chars, block_size, MPI_CHAR, image_chars,recvCounts, displs, MPI_CHAR,0, MPI_COMM_WORLD);
        //MPI_Gatherv(my_image_chars, block_size+2*n, MPI_CHAR, image_chars, sendCounts, displs, MPI_CHAR, 0,MPI_COMM_WORLD);
        MPI_Send(my_image_chars,block_size+2*n, MPI_CHAR, 0,0, MPI_COMM_WORLD);
   int k,p;
    if(my_rank == 0){
        //receive the computed my_image_chars from all processes
        my_new_image_chars = malloc(block_size*sizeof(int));
        new_image_chars = malloc(n*m*sizeof(int));
        for(i=0;i<n*m;i++){
            new_image_chars[i] = 0;
        }
        for(i=0;i<num_procs;i++){
            MPI_Recv(my_new_image_chars,sendCounts[i], MPI_CHAR,i,0,MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            start = displs[i];//i*(sendCounts[i]-2*n);
            for(k=0;k<sendCounts[i];k++){
                new_image_chars[start + k]= my_new_image_chars[k];
            }

        }

        export_JPEG_file(output_jpeg_filename, new_image_chars,m,n,c,75);
    }

    deallocate_image(&u);
    deallocate_image(&u_bar);
    //printf("Hello World!\n");
    MPI_Finalize();
    return 0;
}



void allocate_image(image *u, int m, int n){
    u->image_data = malloc((m+2)*sizeof(double*));
    u->m=m;
    u->n=n;
    int i;
    //printf("in allocate n= %d", n);
    for(i=0;i<m+2;i++){
        u->image_data[i] = malloc(n*sizeof(double));
    }
}
void deallocate_image(image *u){
    int i;
    for(i=0;i<u->m;i++){
        free(u->image_data[i]);
    }
    free(u->image_data);
}
void iso_diffusion_denoising(image *u, image *u_bar, float kappa, int iters){
    int my_rank, num_procs;
    MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);
    MPI_Comm_size(MPI_COMM_WORLD, &num_procs);
    int i,j,k,m,n,l,p, true_m;
    m = u->m;
    true_m = m+2;
    n=u->n;
    double ghost_send[n];
    double ghost_recv[n];



    ////////////////////////
    //denoising alogorithm//
    ////////////////////////

    for(k=0;k<iters;k++){
        //printf("%d", k);
        for(i=1;i<true_m-1; i++){
            for(j=1;j<n-1;j++){
                u_bar->image_data[i][j] = u->image_data[i][j] + kappa*(u->image_data[i-1][j] + u->image_data[i][j-1]-4*u->image_data[i][j] + u->image_data[i][j+1] + u->image_data[i+1][j]);


            }
        }
        /////////////////////////////////
        //here we must send//
        /////////////////////////////////
        //////////
        //Bottom//
        //////////
        if(my_rank!=num_procs-1){
            for(i=0;i<n;i++){
                ghost_send[i] = u_bar->image_data[true_m-2][i];
            }

            MPI_Send(ghost_send,n,MPI_DOUBLE, my_rank+1,0,MPI_COMM_WORLD);



        }

        ///////
        //top//
        ///////
        if(my_rank!=0){
            for(i=0;i<n;i++){
                ghost_send[i] = u_bar->image_data[1][i];
            }

            MPI_Recv(ghost_recv,n, MPI_DOUBLE,my_rank-1, 0, MPI_COMM_WORLD,MPI_STATUS_IGNORE);
            MPI_Send(ghost_send,n, MPI_DOUBLE,my_rank-1,1,MPI_COMM_WORLD);

            for(i=0;i<n;i++){
                u_bar->image_data[0][i] = ghost_recv[i];
            }
        }
        if(my_rank!=num_procs-1){
            MPI_Recv(ghost_recv,n,MPI_DOUBLE,my_rank+1, 1, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            for(i=0;i<n;i++){
                u_bar->image_data[true_m-1][i] = ghost_recv[i];

            }
        }



        for(l=0;l<true_m;l++){
            for(p=0;p<n;p++){
                u->image_data[l][p] = u_bar->image_data[l][p];
            }
        }

    }


}

void convert_char_to_float(unsigned char *pict, image *im, int m, int n, int start){
    int i,x,y;
    for(i=start;i<m*n;i++){
        y = i%n;
        x = (i - y)/n;
        im->image_data[x][y] = (float) (pict[i]);
    }
}

void convert_float_to_char(unsigned char **image_chars, image *u_bar, int m, int n, int start){
    int x,y, pict_number, value;

    //image_chars[m-1][n-1];
    //int si = sizeof(image_chars);
    //printf("%d\n", pict_number);
    //printf("%d\n",si);
    for(x=0;x<m-1;x++){
        for(y=0;y<n;y++){
            pict_number = x*n + y;
            value = (int)(u_bar->image_data[x][y]);
            //printf("%d",pict_number);//(unsigned char) value;
            *image_chars[pict_number] = (unsigned char) value;
        }
    }



//    int x,y, pict_number;
//    float value,max=0,min=1000;
//    for(x=0;x<m;x++){
//        for(y=0;y<n;y++){
//            pict_number = y*n + x;
//            unsigned char buff;
//            value = (im->image_data[x][y]);


//        }
//    }


}

//mpicc -o runoblig oblig1_main.c libsimplejpeg.a



