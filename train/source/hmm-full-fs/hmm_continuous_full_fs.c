/*
********************************************************************************
*                                                                              *
	Name: hmm_continuous_full_fs.c

	Author: Jose

	Date:   march/95

	Description: This program creates a continuous Markov model, using the Baum-Welch (forward-backward) algorithm. This algorithm is described int the paper " A Tutorial on Hidden Markov Models and selected Applications on Speech Recognition " of L. R. Rabiner. ( february of 1989). It is considered full covariance matrix.
        This program must be used to create model for isolated words. The model that represents each word will be stored in a separated file. 

	Inputs: word that will be represented by the model, number of states,
                number of parameters to train the model, number of mixtures,
                names of files with parameters, output file name,
                name of initial model, if there is one.

	Outputs:  model file.
	
*                                                                              * ********************************************************************************
*/



#include <math.h>           
#include <stdio.h>
#include <stdlib.h>
#include <fcntl.h>
#include <time.h>
#include <ctype.h>
#include <string.h>
#include <sys/types.h>
#include <sys/times.h>


#define THRESHOULD  			1.0e-3      /* threshould to finish the training procedure */
#define DELTA  					1           /* maximum difference between index of states where transition is allowed */
#define FINITE_PROBAB 			1.0e-5    	/* minimum value of mixture coefficients and elements of diagonal covariance matrix */
#define MAX_COEF_NUMBER 		9       	/* maximum number of coefficients per vector */
#define MAX_STATES_NUMBER 		20     		/* maximum number of states */
#define MAX_PARAMETERS_NUMBER 	6  			/* maximum number of parameters */
#define MAX_MIXTURE_NUMBER 		3    		/* maximum number of mixtures */
#define MAX_TIME 				500         /* maximum time */
#define MAX_STRING_SIZE			100
#define MAX_WORD_SIZE			50


  
/* global variables */

FILE*	f_in[MAX_PARAMETERS_NUMBER];
FILE*	f_in1;  							 /* input file pointers */
FILE*   f_out;                         		 /* output file pointer */
 
struct mixture
{   	/* mixture */
       double mean[MAX_COEF_NUMBER];       					/* mixture mean vector */
       double cov_matrix[MAX_COEF_NUMBER][MAX_COEF_NUMBER]; /* mixture diagonal covariance matrix */
       double det;                         					/* covariance matrix determinant */
};    /* end of struct mixture */

struct state
{
       double mix_coef[MAX_MIXTURE_NUMBER];    /* mixture coefficients */
       struct mixture mix[MAX_MIXTURE_NUMBER]; /* mixtures */
};   /* end of structure state */


/*  functions */

FILE *opening_file_read( );         /* open file for reading */
FILE *opening_file_write( );        /* open file for writing */
int reading_param_file_name( );     /* read parameter file name */
int reading_coef( );                /* read coefficients */
int reading_coef_number( );         /* read number of coefficients */
void reading_model( );              /* read model */
void creating_initial_model( );     /* create initial model */
void init_transition_probab( );     /* calculate intial transition probabilities */
void init_mix_param( );             /* calculate initial set of mixture parameters */
void init_mix_mean( );              /* create an initial set of mixture means */
void splitting( );                  /* split each mixture mean in 2 new mixture means */
void calc_symbol_probab( );         /* calculate output symbol probability density */
double calc_gaus( );                /* calculate gaussian probability density of each mixture */
double classifying( );              /* classify a vector in a cluster */        
void new_mix_mean( );               /* calculate new mixture means */
void sorting( );                    /* sort index of cells */
void changing_zero_coef( );         /* change mixture coefficients  less than threshould */
void calc_alpha( );                 /* calculate forward variable */
void calc_beta( );                  /* calculate backward variable */
double calc_probability( );         /* calculate probability */
void calc_transition_probab( );     /* calculate transition probabilities */
void calc_den_mix_coef( );          /* calculate variable to calculate mixture coefficients */
void calc_mix_param( );             /* calculate mixture probabilities  */
void updating_transition_probab( ); /* update transition probabilities */
void updating_mix_param( );         /* update mixture parameters */
void decomposition( );              /* decomposes covariance matrix as TDT' */
double calc_det( );                 /* calculate diagonal matrix determinat */
void inv_triang_matrix( );          /* invert triangular matrix */
double inv_cov_matrix( );           /* invert covariance matrix */
void treat_zero_det( );             /* treat mixtures with cov. matrix determinant equal to zero */
void writing_model( );              /* write model */
void writing_text( );               /* write text file about model */



int main (int argc, char *argv[])    /* main  program */
{
	double	transition_probab[MAX_STATES_NUMBER][MAX_STATES_NUMBER]; 			/* transition probabilities  */
	double	symbol_probab[MAX_PARAMETERS_NUMBER][MAX_STATES_NUMBER][MAX_TIME];	/* symbol probabilities */
	double	gaus_probab_dens[MAX_PARAMETERS_NUMBER][MAX_TIME][MAX_STATES_NUMBER][MAX_MIXTURE_NUMBER];	/* gaussian probability density */
	double	alpha[MAX_STATES_NUMBER][MAX_TIME]; /* forward variable */
	double	beta[MAX_STATES_NUMBER][MAX_TIME]; /* backward variable */
	double	scaling_factor[MAX_TIME]; /* scaling factor */
	double	num_trans_probab[MAX_STATES_NUMBER][MAX_STATES_NUMBER]; /* variable to calculate transition probabilities */
	double	den_trans_probab[MAX_STATES_NUMBER]; /* variable to calculate transition probabilities */
	double	den_mixture_coef[MAX_STATES_NUMBER]; /* variable to calculate mixture coefficients */
	double	coef_vector[MAX_COEF_NUMBER]; /* vector of coefficients */
	int		coef_number[MAX_PARAMETERS_NUMBER];  /* number of coefficients per vector */
	int		mixture_number[MAX_PARAMETERS_NUMBER];  /* number of mixtures per state (number of mixtures may be different for each parameter) */
	int		pi[MAX_STATES_NUMBER];   /* initial state probability */
	char	data_file[MAX_PARAMETERS_NUMBER][MAX_STRING_SIZE];     /* data file */
	char	param_file[MAX_PARAMETERS_NUMBER][MAX_STRING_SIZE];    /* parameters files */
	char   	starting_time_f[25];  /* variable to store formatted stating time */
	char   	ending_time_f[25];    /* variable to store formatted stating time */
	char   	cpu_time_f[25];       /* variable to store formatted cpu time  */
	char   	output_file[MAX_STRING_SIZE];     /* output file names */
	char	text_file[MAX_STRING_SIZE];       /* text file name */
	char	initial_model[MAX_STRING_SIZE];   /* name of initial model */
	char	word[MAX_WORD_SIZE];             /* word to create the model */

	struct 	state state_mix[MAX_PARAMETERS_NUMBER][MAX_STATES_NUMBER]; /* array with mixture parameters (considering all states and all kind of parameters ) */

	struct 	state num_mix_param[MAX_PARAMETERS_NUMBER][MAX_STATES_NUMBER]; /* variable  to calculate mixtures parameters */

	double 	probab;               /* total probability in each iteration */
	double	old_probab = 1.0;     /* probability in the last iteration */
	double	probab_variation;     /* probability variation between 2 consecutive iterations */
	double	aux;                  /* auxiliar variable */

	int		param_number;            /* number of parameters  to create the model */
	int		states_number;           /* number of states in the HMM */
	int		obs_time;                /* time (number of frames in each observation sequence) */
	int		iteration = 0;           /* number of iterations */
	int		exemplar_number;         /* number of exemplar in training sequence */

	struct tm *cpu_tm;           /* get process times */

	struct tms aux_time;         /* auxiliar variable to compute cpu time */

	time_t cpu_time;             /* cpu time */

	time_t starting_time;        /* variable to get  starting time  */

	time_t ending_time;          /* variable to get  starting time  */

	register int i,j,k,l,m;       /* variables to control loops */


	time(&starting_time);        /* getting starting time */

	strftime(starting_time_f,100,"%d-%h-%Y %X",localtime(&starting_time));    /* formatting the date (dd-mm-yyyy) and time (hh:mm:ss) */

	/* testing the number of arguments in the command line */
	if (argc < 7)
	{
		puts("Usage: hmm_continuous_full_fs word states_number param_number mix_number1 ... mix_numberN  input_file1 ... input_fileN output_file [initial_model]");
		puts("word: word that will be represented by the model");
		puts("states_number: number of states");
		puts("param_number: number of parameters to train the model");
		puts("mix_number1: number of mixtures per state (parameter 1)");
		puts("mix_numberN: number of mixtures per state (parameter N)");
		puts("input_file1: name of file with names of files with parameters 1");
		puts("input_fileN: name of file with names of files with parameters N");
		puts("output_file: output file name");
		puts("initial_model: name of initial model, if there is one");
		exit(1);
	}

	strncpy(word,argv[1],MAX_WORD_SIZE); 	/* word that will be represented by the model */

	states_number = atoi(argv[2]);     		/* number of states to create the model */

	param_number = atoi(argv[3]);    		/* number of parameters to create the model */

	for (i=0;i<param_number;i++)	mixture_number[i] = atoi(argv[4+i]);   /* number of mixtures to create the model */

	strncpy(output_file,argv[2*param_number+4],MAX_STRING_SIZE);   /* output file name */

	/* creating the name of text file.  The name is equal the name of output file with extension "TXT". This file contains information about the model */
	strncpy(text_file, output_file,MAX_STRING_SIZE);
	strtok(text_file,".");
	strncat(text_file,".txt",MAX_STRING_SIZE);

	/* opening data file */
	for (i=0;i<param_number;i++)
	{
		 strcpy(data_file[i],argv[param_number+4+i]);
		 f_in[i] = opening_file_read(data_file[i],"r");
	}


	if (argc == 2*param_number + 6)
	{ /* testing if there is an initial model */
		strcpy(initial_model,argv[argc]);   /* initial model file name */
		/* reading initial model */
		reading_model(initial_model,&param_number,&states_number,mixture_number,
					  coef_number,transition_probab,state_mix,word);
	}
	else
	{
		/* creating initial model */
		creating_initial_model(param_number,data_file,states_number,mixture_number,
							   coef_number,transition_probab,state_mix);
	}

	/* setting inital states probability */
	/* as the left-right model is being used, the initial state is always the state 0 */
	pi[0] = 1;
	for (i=1;i<states_number;i++)	pi[i] = 0;

	/* creation of a HMM using Forward-Backward algorithm (Baum-Welch) */
	printf("\r\nCreating HMM using Forward-Backward algorithm (Baum-Welch)");
	do
	{
		iteration++;
		exemplar_number = 0;
		probab = 0.0;
		/* setting arrays num_trans_probab and den_trans_probab  to zero */
		for (i=0;i<states_number;i++)
		{
			 for (j=0;j<states_number;j++)	num_trans_probab[i][j] = 0.0;

			 den_trans_probab[i] = 0.0;
			 den_mixture_coef[i] = 0.0;
		}

		/* setting arrays mix_coef, mean and  cov_matrix to zero */
		for (i=0;i<param_number;i++)
		{
			 for (j=0;j<states_number;j++)
			 {
				  for (k=0;k<mixture_number[i];k++)
				  {
					   for (l=0;l<coef_number[i];l++)
					   {
							num_mix_param[i][j].mix[k].mean[l] = 0.0;
							for (m=0;m<coef_number[i];m++)	num_mix_param[i][j].mix[k].cov_matrix[l][m] = 0.0;
					   }
					   num_mix_param[i][j].mix_coef[k] = 0.0;
				  }
			 }
		}
		printf("\r\nStarting training sequence");
		while (reading_param_file_name(f_in,data_file,param_file,param_number) != EOF) /* testing training sequence end */
		{
		   for (i=0;i<param_number;i++)
		   {
			   printf("\r\nOpenning %s",param_file[i]);
			   f_in1 = opening_file_read(param_file[i],"rb"); /* opening data file */
			   obs_time = 0;
			   reading_coef_number(f_in1,param_file[i]);
			   while (reading_coef(f_in1,param_file[i],coef_number[i],	coef_vector) != 0)
			   {
				   calc_symbol_probab(states_number,mixture_number[i],
									   coef_number[i],coef_vector,state_mix[i],gaus_probab_dens[i][obs_time],
									   symbol_probab[i],obs_time);    /* computing symbol probabilities */
				   obs_time++;
			   }   /* end of while */
			   fclose(f_in1);  /* closing data file */
		   }  /* end of for */

		   printf("\r\nCalculating alpha, beta and transitions probabilities...");

		   calc_alpha(states_number,obs_time,param_number,alpha,scaling_factor,
					  transition_probab,symbol_probab,pi);
		   calc_beta(states_number,obs_time,param_number,beta,scaling_factor,
					 transition_probab,symbol_probab);
		   calc_transition_probab(states_number,obs_time,param_number,alpha,
								  beta,scaling_factor,transition_probab,symbol_probab,num_trans_probab,
								  den_trans_probab );
		   calc_den_mix_coef(obs_time,states_number,alpha,beta,scaling_factor,
							 den_mixture_coef);

		   for (i=0;i<param_number;i++)
		   {
				obs_time = 0;
				f_in1 = opening_file_read(param_file[i],"rb"); /* opening data file */
				reading_coef_number(f_in1,param_file[i]);
				while (reading_coef(f_in1,param_file[i],coef_number[i],	coef_vector) != 0)
				{
					   calc_mix_param(obs_time,states_number,
									  mixture_number[i],coef_number[i],coef_vector,alpha,beta,scaling_factor,
									  gaus_probab_dens[i][obs_time],num_mix_param[i],state_mix[i]); /* computing mixture parameters */
					   obs_time++;
				}   /* end of while */
				fclose(f_in1);  /* closing data file */
		   }

		   probab += calc_probability(obs_time,scaling_factor, alpha[states_number-1][obs_time-1]);
		   exemplar_number++;
		}  /* end of while (end of training sequence ) */

		printf("\r\nEnding training sequence");

		/* distortion variation between two consecutive iterations */
		probab_variation =  fabs((old_probab - probab)/old_probab);

		printf("\r\nVerifying Probability: %f > Threshold: %f", probab_variation, THRESHOULD);
		if (probab_variation > THRESHOULD)
		{
			old_probab = probab;
			/* updating transition probabilities */
			updating_transition_probab(states_number,num_trans_probab,
									   den_trans_probab,transition_probab );
			/* updating mixture parameters */
			for (i=0;i<param_number;i++)
			{
				 updating_mix_param(states_number,mixture_number[i],
									coef_number[i],den_mixture_coef, num_mix_param[i],state_mix[i]);
				 if (coef_number[i] > 1)
				 {
					 for (j=0;j<states_number;j++)
					 {
						  for (k=0;k<mixture_number[i];k++)
						  {
							   state_mix[i][j].mix[k].det =  inv_cov_matrix(coef_number[i],state_mix[i][j].mix[k].cov_matrix);
						  }
						  treat_zero_det(coef_number[i],mixture_number[i],&(state_mix[i][j]));
					 }
				 }
				 else
				 {
					 for (j=0;j<states_number;j++)
					 {
						 for (k=0;k<mixture_number[i];k++)
						 {
							 state_mix[i][j].mix[k].det =  state_mix[i][j].mix[k].cov_matrix[0][0];
							 state_mix[i][j].mix[k].cov_matrix[0][0] = 1.0/state_mix[i][j].mix[k].cov_matrix[0][0];
						 }
					 }
				}
			 }
			 for (i=0;i<param_number;i++)	rewind(f_in[i]);
		}
		/* end of do-while */
	}while ( probab_variation > THRESHOULD); /* comparing the probability variation with the threshould to finish model creation */

	printf("\r\nFinal Probability = %f\r\n\r\n",probab_variation);

	probab /= (double) exemplar_number;

	/* computing cpu time */
	times(&aux_time);       /* getting cpu time */
	aux = (aux_time.tms_utime )/60.0;
	cpu_time = (int) aux;
	cpu_tm = gmtime(&cpu_time);
	cpu_tm->tm_mday -=  1;
	strftime(cpu_time_f,50,"%d %X",cpu_tm);  /* formatting cpu time (dd hh:mm:ss) */

	time(&ending_time);        /* getting starting time */

	strftime(ending_time_f,100,"%d-%h-%Y %X",localtime(&ending_time));    /* formatting the date (dd-mm-yyyy) and time (hh:mm:ss) */

	for (i=0;i<param_number;i++)	fclose(f_in[i]);          /* closing data files */

	f_out = opening_file_write(output_file,"wb");
	writing_model(output_file,states_number,param_number,mixture_number,
	coef_number,transition_probab,state_mix,word);
	fclose(f_out);

	f_out = opening_file_write(text_file,"w");
	writing_text(text_file,output_file,word,states_number,param_number,
	mixture_number,data_file,starting_time_f,ending_time_f,cpu_time_f,
	exemplar_number,probab,iteration  );
	fclose(f_out);

	return 0;
}   /* end of main */






/*
********************************************************************************
*                                                                              *
*   opening_file_read:      				                       *
*                                                                              *
*   This function opens a file for reading.                                    *
*									       *
*   input: file_name, mode.						       *
*									       *
*   output: f_in.       						       *
*									       *
********************************************************************************
*/
 
FILE* opening_file_read(char* file_name,   /* pointer to file name */
						char* mode        /* pointer to reading mode */
						)
{
	FILE *f_in;        /* pointer to reading file */

	if ((f_in = fopen(file_name,mode)) == NULL)
	{
		 printf("file %s not found \n",file_name);
		 exit(1);
	}

	return(f_in);

}      /* end of function opening_file_read */




/*
********************************************************************************
*                                                                              *
*   opening_file_write:      				                       *
*                                                                              *
*   This function opens the file for writing.                                  *
*									       *
*   input: file_name, mode.						       *
*									       *
*   output: f_out.       						       *
*									       *
********************************************************************************
*/
 
FILE *opening_file_write(file_name,mode)
char *file_name;   /* pointer to file name */
char *mode;        /* pointer to writing mode */

{
FILE *f_out;       /* pointer to writing file */


if((f_out=fopen(file_name, mode)) == NULL) {
   printf("can't open file %s \n",file_name);
   exit(1);
}

return(f_out);

}      /* end of function opening_file_write */






/*
********************************************************************************
*                                                                              *
*   reading_param_file_name:          		                               *
*                                                                              *
*   This function reads names of parameters files.                             *
*									       *
*   input: f_in, data_file, param_number.          			       *
*									       *
*   output: param_file.          					       *
*									       *
********************************************************************************
*/
 

int reading_param_file_name(FILE*	f_in[MAX_PARAMETERS_NUMBER],           /* pointer to input file */
							char 	data_file[MAX_PARAMETERS_NUMBER][100],   /* pointer to input file name */
							char 	param_file[MAX_PARAMETERS_NUMBER][100],  /* pointer to input file name */
							int 	param_number                            /* number of parameters */
							)
{
	int aux;              /* end of file detector */
	register int i;

	for (i=0;i<param_number;i++)
	{
		 aux = fscanf(f_in[i],"%s",param_file[i]);
		 if (ferror(f_in[i]))
		 {
			 printf("reading error on file %s \n",data_file[i]);
			 exit(1);
		 }
	}

	return(aux);

}  /* end of function reading_param_file_name*/









/*
********************************************************************************
*                                                                              *
*   reading_coef:          				                       *
*                                                                              *
*   This function reads a vector of coefficients.                              *
*									       *
*   input: f_in, file_name, coef_number.				       *
*									       *
*   output: coef_vector, aux.    					       *
*									       *
********************************************************************************
*/
 

int reading_coef(
FILE *f_in,           /* pointer to input file */
char *file_name,      /* pointer to input file name */
int coef_number,      /* number of coefficients */
double *coef_vector  /* pointer to vector of coefficients */
)
{
	int aux;              /* number of bytes read */

	aux = fread(coef_vector,sizeof(double),coef_number,f_in); /* reading coefficients */
	if (ferror(f_in))
	{
		printf("reading error on file %s \n",file_name);
		exit(1);
	}
	return(aux);

}  /* end of function reading_coef */






/*
********************************************************************************
*                                                                              *
*   reading_coef_number:          				               *
*                                                                              *
*   This function reads the number of coefficients per frame.                  *
*									       *
*   input: f_in, file_name.		                 		       *
*									       *
*   output: coef_number.         					       *
*									       *
********************************************************************************
*/
 
int reading_coef_number(FILE *f_in,           /* pointer to input file */
						char *file_name      /* pointer to input file name */
						)
{
	int coef_number=0;      /* number of coefficients */

	fread(&coef_number,sizeof(int),1,f_in); /* reading number of coefficients */
	if (ferror(f_in))
	{
		printf("reading error on file %s \n",file_name);
		exit(1);
	}
	return(coef_number);

}  /* end of function reading_coef */







/*
********************************************************************************
*                                                                              *
*   reading_model:                       		                       *
*                                                                              *
*   This function reads a model.                                               *
*									       *
*   input: model_name.                                                         *
*									       *
*   output: param_number, states_number, mixture_number, coef_number,          *
*           transition_probab, state_mix, word.                                *
*									       *
********************************************************************************
*/

void reading_model(
	char *model_name,       /* name of model */
	int *param_number,      /* number of parameters */
	int *states_number,     /* number of states */
	int *mixture_number,    /* number of mixtures */
	int *coef_number,       /* number of coefficients */
	double transition_probab[MAX_STATES_NUMBER][MAX_STATES_NUMBER],  /* transition probabilities */
	struct state state_mix[MAX_PARAMETERS_NUMBER][MAX_STATES_NUMBER], /* mixture parameters */
	char *word             /* word */
				)
{
	FILE*	f_in;
	size_t	length;
	int register i,j,k,l;


	/* opening  model file */
	f_in = opening_file_read(model_name,"rb");

	fread(&length, sizeof(size_t),1,f_in);   /* reading length of word */
	if (ferror(f_in))
	{
		printf("reading error on file %s \n",model_name);
		exit(1);
	}

	fread(word, sizeof(char),length,f_in);   /* reading word */
	if (ferror(f_in))
	{
		printf("reading error on file %s \n",model_name);
		exit(1);
	}

	fread(states_number, sizeof(int),1,f_in);   /* reading number of states */
	if (ferror(f_in))
	{
		printf("reading error on file %s \n",model_name);
		exit(1);
	}


	fread(param_number, sizeof(int),1,f_in);   /* reading number of parameters */
	if (ferror(f_in))
	{
		printf("reading error on file %s \n",model_name);
		exit(1);
	}


	fread(mixture_number,sizeof(int),*param_number,f_in);   /* reading number of mixtures */
	if (ferror(f_in))
	{
		printf("reading error on file %s \n",model_name);
		exit(1);
	}

	fread(coef_number,sizeof(int),*param_number,f_in);   /* reading number of coefficients */
	if (ferror(f_in))
	{
		printf("reading error on file %s \n",model_name);
		exit(1);
	}

	for (i=0;i<*states_number;i++)
	{
		 fread(transition_probab[i], sizeof(double),*states_number,f_in);   /* reading initial model - transition probabilities */
		 if (ferror(f_in))
		 {
			 printf("reading error on file %s \n",model_name);
			 exit(1);
		 }
	}

	/* reading mixture parameters */
	for (i=0;i<*param_number;i++)
	{
		 for (j=0;j<*states_number;j++)
		 {
			  /* reading mixture coefficients */
			  fread(state_mix[i][j].mix_coef,sizeof(double),mixture_number[i],f_in);
			  if (ferror(f_in))
			  {
				  printf("reading error on file %s \n",model_name);
				  exit(1);
			  }

			  for (k=0;k<mixture_number[i];k++)
			  {
				   /* reading mixture mean */
				   fread(state_mix[i][j].mix[k].mean,sizeof(double),coef_number[i],	f_in);
				   if (ferror(f_in))
				   {
					   printf("reading error on file %s \n",model_name);
					   exit(1);
				   }

				   /* reading covariance matrix determinant */
				   fread(&(state_mix[i][j].mix[k].det),sizeof(double),1,f_in);
				   if (ferror(f_in))
				   {
					   printf("reading error on file %s \n",model_name);
					   exit(1);
				   }

				   /* reading  inverse covariance matrix (full) */
				   for (l=0;l<coef_number[i];l++)
				   {
						fread(state_mix[i][j].mix[k].cov_matrix[l],sizeof(double),coef_number[i],f_in);
						if (ferror(f_in))
						{
							printf("reading error on file %s \n",model_name);
							exit(1);
						}
				   }
			  }
		 }
	}

	fclose(f_in);    /* closing  initial model file */

}    /* end of function reading_model */






/*
********************************************************************************
*                                                                              *
*   creating_initial_model:             		                       *
*                                                                              *
*   This function creates an initial model.                                    *
*									       *
*   input: param_number, data_file, states_number, mixture_number.             *
*									       *
*   output: coef_number,transition_probab, state_mix.                          *
*									       *
********************************************************************************
*/

void creating_initial_model (
	int param_number,    /* number of parameters */
	char data_file[MAX_PARAMETERS_NUMBER][100],  /* name of file */
	int states_number,   /* number of states */
	int *mixture_number, /* number of mixtures */
	int *coef_number,    /* number of coefficients */
	double transition_probab[MAX_STATES_NUMBER][MAX_STATES_NUMBER],  /* transition probabilities */
	struct state state_mix[MAX_PARAMETERS_NUMBER][MAX_STATES_NUMBER] /* mixture parameters */
							)
{
	register int i;

	/* creating initial set of transition probability matrix using uniform distribuition */
	init_transition_probab(states_number,transition_probab);

	/* creating the inital set of mixture parameters */
	for (i=0;i<param_number;i++)
	{
		 init_mix_param(states_number,mixture_number[i],&coef_number[i], data_file[i],state_mix[i]);
	}

}   /* end of function creating_initial_model */





/*
********************************************************************************
*                                                                              *
*   init_transition_probab:                   		                       *
*                                                                              *
*   This function creates an initial set of transition probabilities.          *
*									       *
*   input: states_number.                                                      *
*									       *
*   output: transition_probab.                                                 *
*									       *
********************************************************************************
*/

void init_transition_probab(int states_number,    /* number of states */
							double transition_probab[MAX_STATES_NUMBER][MAX_STATES_NUMBER]  /* transition probabilities */
							)
{
	register int i,j;

	/* initializing transition probability matrix using uniform distribuition */
	/* Only transitions from state i to states i, i+1,..., i+DELTA are allowed. */
	for (i=0;i<states_number;i++)
	{
		 for (j=0;j<states_number;j++)
		 {
			  if ((j>DELTA+i) || (j<i))					transition_probab[i][j] = 0;
			  else
				  if ((DELTA+1) > (states_number-i))	transition_probab[i][j] = 1.0/ (double) (states_number-i);
				  else									transition_probab[i][j] = 1.0/ (double) (DELTA+1);
		 } /* end of for (j) */
	} /* end of for (i) */

}  /* end of function init_transition_probab */




/*
********************************************************************************
*                                                                              *
*   init_mix_param:                       		                       *
*                                                                              *
*   This function creates an initial set of mixture parameters.                *
*									       *
*   input: states_number, mixture_number,coef_number,training_file, ext.       *
*									       *
*   output: state_mix.                                                         *
*									       *
********************************************************************************
*/

void init_mix_param(
	int states_number,      /* number of states */
	int mixture_number,     /* number of mixtures */
	int *coef_number,       /* number of coefficients */
	char *data_file,        /* file with names of data files */
	struct state state_mix[MAX_STATES_NUMBER] /* structure with mixture parameters */
	                )
{
	FILE*	f_in;
	FILE*	f_in1;

	double	coef_vector[MAX_TIME][MAX_COEF_NUMBER];     /* vector of coefficients */
	double 	mean[MAX_STATES_NUMBER][MAX_MIXTURE_NUMBER][MAX_COEF_NUMBER]; /* mixture mean */
	double 	dif[MAX_COEF_NUMBER];         /* auxiliar variable  to calculate  covariance matrix */

//	int observation_seq[MAX_TIME];  /* observation vector */
	int		state_duration[MAX_STATES_NUMBER];  /* total number of symbols per state */

	char 	param_file[100];

//	double 	aux;  /* auxiliar variable */
	double	sum;

//	int obs_time;                   /* time ( number of frames in  each observation sequence ) */
	int		count_time;                /* symbols counter */
	int		symbol_state;               /* number of symbols per state */
//	int		aux1 = 0;                   /* auxiliar variables */
//	int		aux2 = 0;
	int		index;
	int		remainder;
	int		begin;
	int		end;

	register int i,j,k,l;


	/* creating the initial set of mixture parameters */

	init_mix_mean(states_number,mixture_number,coef_number,data_file,mean);

	f_in1 = opening_file_read(data_file,"r");

	for (i=0;i<states_number;i++)
	{
		 state_duration[i] = 0;
		 for (j=0;j<mixture_number;j++)
		 {
			  state_mix[i].mix_coef[j] = 0.0;
			  for (k=0;k<*coef_number;k++)
				   for (l=0;l<*coef_number;l++)
						state_mix[i].mix[j].cov_matrix[k][l] = 0.0;
		 }
	}

	while (fscanf(f_in1,"%s",param_file) != EOF)
	{ /* testing the end training sequence */

		   f_in = opening_file_read(param_file,"rb"); /* opening data file */
		   count_time = 0;
		   reading_coef_number(f_in,param_file);
		   while (reading_coef(f_in,param_file,*coef_number,coef_vector[count_time]) != 0)
		   {
			   count_time++;
		   }
		   fclose(f_in);  /* closing data file */

		   symbol_state = count_time/states_number;
		   remainder = count_time%states_number;
		   end = 0;
		   for (k=0;k<states_number;k++)
		   {
				begin = end;
				if (k < remainder)
					end += symbol_state +1;
				else
					end += symbol_state;
				for (j=begin;j<end;j++)
				{
					 classifying(coef_vector[j],mixture_number,*coef_number,mean[k],&index);
					 for (l=0;l<*coef_number;l++)	dif[l] = coef_vector[j][l] - mean[k][index][l];

					 for (i=0;i<*coef_number;i++)
					 {
						  for (l=i;l<*coef_number;l++)	state_mix[k].mix[index].cov_matrix[i][l] += dif[i]*dif[l];
					 }
					 state_mix[k].mix_coef[index]++;
				}
				state_duration[k] += end - begin;
		   }

	}  /* end of while  ( testing the EOF) */


	fclose(f_in1);

	for (i=0;i<states_number;i++)
	{
		 for (j=0;j<mixture_number;j++)
		 {
			  for (k=0;k<*coef_number;k++)
				   for (l=k;l<*coef_number;l++)
				   {
					   state_mix[i].mix[j].cov_matrix[k][l] /= state_mix[i].mix_coef[j];
				   }

			  for (k=0;k<*coef_number;k++)
				   if (state_mix[i].mix[j].cov_matrix[k][k] < FINITE_PROBAB)	state_mix[i].mix[j].cov_matrix[k][k] = FINITE_PROBAB;

			  if (*coef_number > 1)
			  {
				  for (k=1;k<*coef_number;k++)
				  {
					   for (l=0;l<k;l++)	state_mix[i].mix[j].cov_matrix[k][l] =  state_mix[i].mix[j].cov_matrix[l][k];
				  }
				  state_mix[i].mix[j].det = inv_cov_matrix(*coef_number,state_mix[i].mix[j].cov_matrix);
			 }
			 else
			 {
				   state_mix[i].mix[j].det = state_mix[i].mix[j].cov_matrix[0][0];
				   state_mix[i].mix[j].cov_matrix[0][0] = 1.0/state_mix[i].mix[j].cov_matrix[0][0];
			 }
			 for (k=0;k<*coef_number;k++)	state_mix[i].mix[j].mean[k] = mean[i][j][k];

		 }
	}

	for (i=0;i<states_number;i++)
	{
		 sum = 0.0;
		 for (j=0;j<mixture_number;j++)
		 {
			  state_mix[i].mix_coef[j] /= (double) state_duration[i];
			  sum += state_mix[i].mix_coef[j];
		 }
		 if (sum > 1.001 || sum < 0.999)
			 printf("error on computing  initial output symbol probabilities: sum = %f \n",sum);
	}

	/* changing values of mixture coefficients  less than a threshould  */
	for (i=0;i<states_number;i++)
		 changing_zero_coef(mixture_number,state_mix[i].mix_coef);

}   /* end of function init_mix_param */



/*
********************************************************************************
*                                                                              *
*   init_mix_mean:                       		                       *
*                                                                              *
*   This function creates an initial set of mixture mean.                      *
*									       *
*   input: states_number, mixture_number, coef_number, training_file, ext.     *
*									       *
*   output: mean.                                                              *
*									       *
********************************************************************************
*/

void init_mix_mean(
	int states_number,      /* number of states */
	int mixture_number,     /* number of mixtures */
	int *coef_number,       /* number of coefficients */
	char *data_file,        /* file with names of data files */
	double mean[MAX_STATES_NUMBER][MAX_MIXTURE_NUMBER][MAX_COEF_NUMBER]  /* mixture mean */
				  )
{
	FILE*	f_in;
	FILE*	f_in1;

	double 	coef_vector[MAX_TIME][MAX_COEF_NUMBER];     /* vector of coefficients */
	double 	sum_vector[MAX_STATES_NUMBER][MAX_MIXTURE_NUMBER][MAX_COEF_NUMBER];
	double 	distortion[MAX_STATES_NUMBER][MAX_MIXTURE_NUMBER];  /* distortion */

	int 	vector_number[MAX_STATES_NUMBER][MAX_MIXTURE_NUMBER];  /* number of vectors in each mixture */

	char 	param_file[MAX_STRING_SIZE];    /* parameter file name */

	int 	obs_time;            /* time ( number of frames in  each observation sequence ) */
	int		old_mix_number;      /* old mixture number */
	int		mix_number;          /* auxiliar mixture number */
	int		index;               /* index of mixture */
//	int		count_time ;         /* symbols counter */
	int		symbol_state;        /* number of symbols per state */
	int		remainder;
	int		begin;               /* auxiliar variables */
	int		end;
	int		ite;

	register int i,j,k,l;


	/* creating the inital set of mixture mean */

	f_in1 = opening_file_read(data_file,"r");

	for (i=0;i<states_number;i++)
	{
		 vector_number[i][0] = 0;
		 for (j=0;j<mixture_number;j++)
			  for (k=0;k<MAX_COEF_NUMBER;k++)
				   mean[i][j][k] = 0.0 ;
	}


	while (fscanf(f_in1,"%s",param_file) != EOF)/* testing end of training sequence */
	{
		f_in = opening_file_read(param_file,"rb"); /* opening data file */

	   *coef_number = reading_coef_number(f_in,param_file);
	   obs_time = 0;
	   while (reading_coef(f_in,param_file,*coef_number,coef_vector[obs_time]) != 0)
	   {
			  obs_time++;
	   }
	   fclose(f_in);  /* closing data file */

	   symbol_state = obs_time/states_number;
	   remainder = obs_time%states_number;
	   end = 0;
	   for (k=0;k<states_number;k++)
	   {
			begin = end;
			if (k < remainder)
				end += symbol_state +1;
			else
				end += symbol_state;

			/* computing the initial mixture mean in each state */
			for (j=begin;j<end;j++)
			{
				for (l=0;l<*coef_number;l++)
				{
					mean[k][0][l] += coef_vector[j][l];
				}
				vector_number[k][0]++;
			}
	   }
	}  /* end of while  ( testing the EOF) */

	rewind(f_in1);

	for (i=0;i<states_number;i++)
	{
		for (l=0;l<*coef_number;l++)
		{
			mean[i][0][l] /= (double) vector_number[i][0];
		}
	 }
	old_mix_number = 1;

	while (old_mix_number < mixture_number)
	{

		/* splitting  mixture means  in two new mixture means */
		for (i=0;i<states_number;i++)
		{
			splitting(*coef_number,old_mix_number,mixture_number,&mix_number,
					  vector_number[i],mean[i],distortion[i]);
		}
		old_mix_number = mix_number; /* saving  mixture number */

		for (ite=0;ite<5;ite++)
		{

			for (k=0;k<states_number;k++)
			{
			   for (i=0;i<old_mix_number;i++)
		   	   {
				   vector_number[k][i] = 0;
				   distortion[k][i] = 0.0;
				   for (j=0;j<*coef_number;j++)
				   {
					   sum_vector[k][i][j] = 0.0;
				   }
		   	   }
			}
			while (fscanf(f_in1,"%s",param_file) != EOF) /* testing the end training sequence */
			{
				f_in = opening_file_read(param_file,"rb"); /* opening data file */
				obs_time = 0;
				reading_coef_number(f_in,param_file);
				while(reading_coef(f_in,param_file,*coef_number,coef_vector[obs_time]) !=0)
				{
					obs_time++;
				}
				fclose(f_in);  /* closing data file */

				symbol_state = obs_time/states_number;
				remainder = obs_time%states_number;
				end = 0;
				for (k=0;k<states_number;k++)
				{
					begin = end;
					if (k < remainder)
						end += symbol_state +1;
					else
						end += symbol_state;

					/* computing the mean of each mixture in each state */
					for (j=begin;j<end;j++)
					{
						distortion[k][index] += classifying(coef_vector[j],old_mix_number,*coef_number,mean[k],&index);
						vector_number[k][index]++;
						for (l=0;l<*coef_number;l++)
						{
							sum_vector[k][index][l] += coef_vector[j][l];
						}
					}
				}

			}  /* end of while  ( testing the EOF) */

			/* computing new mean for each mixture */
			for (i=0;i<states_number;i++)
				 new_mix_mean(*coef_number,old_mix_number,sum_vector[i],
							  vector_number[i],mean[i],distortion[i]);

			rewind(f_in1);
	   }  /* end of for (number of iterations) */

	}  /* end of while  (mixture number) */

}  /* end of function init_mix_mean */








/*
********************************************************************************
*                                                                              *
*   splitting:           				                       *
*                                                                              *
*   This function splits each mixture mean in two new mixture means.           *
*									       *
*   input: coef_number, old_mix_number,mean,mixture_number, mix_number,        *
*          vector_number.                                		       *
*									       *
*   output: mean.                                			       *
*									       *
********************************************************************************
*/
 
void splitting(
	int coef_number,     /* number of coefficients */
	int old_mix_number,  /* old mixture number */
	int mixture_number,  /* mixture number */
	int *mix_number,     /* auxiliar mixture number */
	int *vector_number,  /* number of vectors in each cell */
	double mean[MAX_MIXTURE_NUMBER][MAX_COEF_NUMBER], /* mixture mean */
	double *distortion  /* distortion of each cell */
			)
{
	int index[MAX_MIXTURE_NUMBER];  /* index of each cell */
	int register i,k,l;
	int dif;                        /* auxiliar variable */

	if (2*old_mix_number < mixture_number)
	{
		for (k=0;k<old_mix_number;k++)
		{
			 for (l=0;l<coef_number;l++)	mean[old_mix_number+k][l] = mean[k][l]*(1.05);
			 for (l=0;l<coef_number;l++)	mean[k][l] = mean[k][l]*(0.95);
		}
		*mix_number = 2*old_mix_number;
	}
	else
	{
		  /* sorting the cells in decreasing  order considering distortion of  each cell */
		  sorting(distortion,index,old_mix_number);
		  dif = mixture_number - old_mix_number;
		  for (k=0;k<dif;k++)
		  {
			  i = index[k];
			  for (l=0;l<coef_number;l++)
			  {
				  mean[old_mix_number+k][l] = mean[i][l]*(1.005);
			  }
			  for (l=0;l<coef_number;l++)
			  {
				  mean[i][l] = mean[i][l]*(0.995);
			  }
		  }
		  *mix_number = dif + old_mix_number;
	}
  
}  /* end of function splitting */






/*
********************************************************************************
*                                                                              *
*   classifying:           				                       *
*                                                                              *
*   This function classifies a vector in a cluster.                            *
*									       *
*   input: coef_vector, mix_number, coef_number, mean.            	       *
*									       *
*   output: index, min_distance.                      			       *
*									       *
********************************************************************************
*/
 
double classifying(
	double  *coef_vector,           /* vector of coefficients */
	int mix_number,                 /* number of mixtures */
	int coef_number,                /* number of coefficients */
	double mean[][MAX_COEF_NUMBER], /* mean */
	int *index                     /* index of cluster in wich the vector was classified */
				  )
{
	double  distance;               /* distance to classify each vector                                                                    in each cluster */
	double	min_distance;           /* minimum distance */
	double	aux;                    /* auxiliar variable */

	register int i,j;

	/* classifying the vectors in the clusters */

	min_distance = 1.0e20;

	for (i=0;i<mix_number;i++)
	{
		 distance = 0.0;

		 /* computing the distance between input vector and centroid of cluster i */
		 for (j=0;j<coef_number;j++)
		 {
			  aux = mean[i][j] - coef_vector[j];
			  distance += aux*aux;
		 } /* end of for(j) - distance computation */

		 if (distance < min_distance)
		 { /* computing the minimum distance */
			 min_distance = distance;
			 *index = i;
		 } /* end of if (distance < min_distance) */

	} /* end of for(i) */

	return(min_distance);

}  /* end of function classifying */





/*
********************************************************************************
*                                                                              *
*   new_mix_mean:           				                       *
*                                                                              *
*   This function creates a new set of mixture means.                          *
*									       *
*   input: coef_number, mix_number, sum_vector, vector_number. 		       *
*									       *
*   output: mean.                                			       *
*									       *
********************************************************************************
*/
 

void new_mix_mean(int coef_number,    /* number of coefficients */
				  int mix_number,     /* mixture number */
				  double sum_vector[MAX_MIXTURE_NUMBER][MAX_COEF_NUMBER],  /* sum of vectors in each mixture */
				  int *vector_number, /* number of vectors in each mixture */
				  double mean[MAX_MIXTURE_NUMBER][MAX_COEF_NUMBER],  /* mixture mean */
				  double *distortion /* distortion of each cell */)
{
	int index[MAX_MIXTURE_NUMBER];
	int register i,j,k,l;


	for (j=0;j<mix_number;j++)
	{
		 for (k=0;k<coef_number;k++)
			  mean[j][k] = sum_vector[j][k]/(double) vector_number[j];
	}

	/* treating empty cells */
	sorting(distortion,index,mix_number);  /* sorting cells in decreasing order considering distortion of each cell */
	i = 0;
	for (j=0;j<mix_number;j++)
	{
		 if (vector_number[j] == 0)
		 {
			 l = index[i++];
			 for (k=0;k<coef_number;k++)	mean[j][k] = mean[l][k]*(1.005);
			 for (k=0;k<coef_number;k++)	mean[l][k] = mean[l][k]*(0.995);
		 }
	}
}  /* end of function new_mix_mean */




/*
********************************************************************************
*                                                                              *
*   sorting:             				                       *
*                                                                              *
*   This function sorts cells in decreasing order considering the number of    *
*   vectors in each cell.                                                      *
*									       *
*   input: vector_number, index, mixture_number.                      	       *
*									       *
*   output: index.                                                             *
*									       *
********************************************************************************
*/

void sorting(double *vector,	/* number of vector in each cell */
			 int *index,          /* index of each cell */
			 int mixture_number  /* number of mixture */)
{
	int done = 0;      /* auxiliar variable */
	register int i,j,k;


	for (i=0;i<mixture_number;i++) index[i] = i;

	while (!done)
	{
		done = 1;
		for (i=0;i<(mixture_number-1);i++)
		{
			j = index[i];
			k = index[i+1];
			if (vector[j] < vector[k])
			{
				index[i] = k;
				index[i+1] = j;
				done = 0;
			}  /* end of if */
		}  /* end of for */
	}  /* end of while */
}    /* end of function sorting */





/*
********************************************************************************
*                                                                              *
*   changing_zero_coef:                        		                       	   *
*                                                                              *
*   This function changes the values of mixture coefficients less than a       *
*   threshould to values equal a threshould.                                   *
*									       									   *
*   input: mixture_number, mixture_coef.                                       *
*									       *
*   output: mixture_coef.                                                      *
*									       *
********************************************************************************
*/

void changing_zero_coef(int mixture_number,   /* number of mixtures */
						double *mixture_coef /* mixture coefficients */)
{
	double sum = 0.0;     /* auxiliar variable */
	register int k;

	/* changing values of mixture coefficients less than a threshould */
	for (k=0;k<mixture_number;k++)
	{
		 if (mixture_coef[k] < FINITE_PROBAB)
		 {
			 mixture_coef[k] = FINITE_PROBAB;
		 }
		 sum += mixture_coef[k];
	}
	for (k=0;k<mixture_number;k++)	mixture_coef[k] /= sum;
}   /* end of function changing_zero_coef */





/*
********************************************************************************
*                                                                              *
*   calc_alpha:                          		                       *
*                                                                              *
*   This function calculates the forward variable (alpha).                     *
*									       *
*   input: states_number,obs_time, param_number, transition_probab,            *
*          symbol_probab,pi.                                                   *
*									       *
*   output: alpha, scaling_factor.                                             *
*									       *
********************************************************************************
*/

void calc_alpha(int states_number,       /* number of states */
				int obs_time,            /* number of symbols */
				int param_number,        /* number of parameters */
				double alpha[MAX_STATES_NUMBER][MAX_TIME],  /* forward variable - alpha */
				double *scaling_factor,  /* scaling factor */
				double transition_probab[MAX_STATES_NUMBER][MAX_STATES_NUMBER],  /* transition probabilities */
				double symbol_probab[MAX_PARAMETERS_NUMBER][MAX_STATES_NUMBER][MAX_TIME],
				int *pi                 /* output symbol probabilities */)
{
	double 		product;
	double	   	sum;
	double	   	aux;
//	int 		symbol;             /* auxiliar variables */
	register int i,j,k;


	/* computing forward variable alpha */

	/* inicialization */
	sum = 0.0;
	for (i=0;i<states_number;i++)
	{
		 product = 1.0;
		 for (j=0;j<param_number;j++)
		 {
			  product *=  symbol_probab[j][i][0];
//			  printf("\nsymbol_probab[%d][%d][0]: %f",j,i,symbol_probab[j][i][0]);
		 }
		 alpha[i][0] = pi[i]*product;
		 sum += alpha[i][0];
//		 printf("\nalpha[%d][0]: %f",i,alpha[i][0]);
	}

	scaling_factor[0] = 1.0/sum; /* scaling factor */

	for (i=0;i<states_number;i++)  /* scaling alpha[i][0] */
		 alpha[i][0] *= scaling_factor[0];

	/* induction */
	for (k=1;k<obs_time;k++)
	{
		 sum = 0.0;
		 for (i=0;i<states_number;i++)
		 {
			  aux = 0.0;
			  for (j=0;j<states_number;j++)	aux += alpha[j][k-1]*transition_probab[j][i];
			  product = 1.0;
			  for (j=0;j<param_number;j++)
			  {
				   product *= symbol_probab[j][i][k];
			  }
			  alpha[i][k] = aux*product;
			  sum += alpha[i][k];
		 }
		 scaling_factor[k] = 1.0/sum;
//		 printf("\nscaling_factor[%d]: %f",k, scaling_factor[k]);

		 for (i=0;i<states_number;i++)	alpha[i][k] *= scaling_factor[k];  /* scaling alpha[i][k] */
	}
}   /* end of calc_alpha */




/*
********************************************************************************
*                                                                              *
*   calc_beta:                          		                       *
*                                                                              *
*   This function calculates the backward variable ( beta).                    *
*									       *
*   input: states_number,obs_time, param_number,            	               *
*          scaling_factor, transition_probab, symbol_probab.                   *
*									       *
*   output: beta.                                                              *
*									       *
********************************************************************************
*/

void calc_beta(int states_number,       /* number of states */
			   int obs_time,            /* number of symbols */
			   int param_number,        /* number of parameters */
			   double beta[MAX_STATES_NUMBER][MAX_TIME],  /* backward variable  - beta */
			   double *scaling_factor,  /* scaling factor */
			   double transition_probab[MAX_STATES_NUMBER][MAX_STATES_NUMBER],   /* transition probabilities */
			   double symbol_probab[MAX_PARAMETERS_NUMBER][MAX_STATES_NUMBER][MAX_TIME]  /* output symbol probabilities */)
{
	double 	product;
//	double  sum;
	double  aux;
//	int 	symbol;             /* auxiliar variables */
	register int i,j,k,l;

	/* computing backward variable beta */

	/* initilization */

	for (i=0;i<states_number-1;i++) beta[i][obs_time-1] = 0.0;

	beta[states_number-1][obs_time-1] = 1.0;   /* final state */

	/* scaling */
	beta[states_number-1][obs_time-1] *= scaling_factor[obs_time-1];

	/* induction */
	for (k=obs_time-2;k>=0;k--)
	{
		for (i=0;i<states_number;i++)
		{
			aux = 0.0;
			for (j=0;j<states_number;j++)
			{
				product = 1.0;
				for (l=0;l<param_number;l++)
				{
					product *= symbol_probab[l][j][k+1];
				}
				aux += beta[j][k+1]*transition_probab[i][j]*product;
			}
			beta[i][k] = aux;
		 }

		/* scaling */
		for (i=0;i<states_number;i++)
		{
			 beta[i][k] *= scaling_factor[k];
			 if (isinf(beta[i][k]) == 1) beta[i][k] = 1.0e200;
		}
	}
}    /* end of calc_beta */






/*
********************************************************************************
*                                                                              *
*   calc_probability:                   		                       *
*                                                                              *
*   This function calculates the probability.                                  *
*									       *
*   input: obs_time, scaling_factor, alpha.           	                       *
*									       *
*   output: probability.                                                       *
*									       *
********************************************************************************
*/

double calc_probability(int		obs_time,            /* time (equal to number of frames) */
						double*	scaling_factor,  /* scaling factor */
						double	alpha)           /* alpha[states_number-1][obs_time-1] */
{
	double probability = 0.0;    /* probability */
	register int i;


//	printf("\nalpha: %f",alpha);

	for (i=0;i<obs_time;i++)
	{
		probability -= log(scaling_factor[i]);
//		printf("\nscaling_factor: %f",scaling_factor[i]);
//		printf("\nprobability: %f",probability);
	}

	probability += log(alpha);
//	printf("\nprobability+log(alpha): %f",probability);

	return(probability);

}  /* end of calc_probability */







/*
********************************************************************************
*                                                                              *
*   calc_transition_probab:                   		                       *
*                                                                              *
*   This function calculates the transition probabilities.                     *
*									       *
*   input: states_number, obs_time, param_number, alpha, beta,                 *
*          scaling_factor, transition_probab, symbol_probab.                   *
*									       *
*   output: num_trans_probab, den_trans_probab.                                *
*									       *
********************************************************************************
*/

void calc_transition_probab(int states_number,        /* number of states */
							int obs_time,             /* number of symbols */
							int param_number,         /* number of parameters */
							double alpha[MAX_STATES_NUMBER][MAX_TIME], /* forward variable - alpha */
							double beta[MAX_STATES_NUMBER][MAX_TIME],   /* backward variable - beta */
							double *scaling_factor,    /* scaling factor */
							double transition_probab[MAX_STATES_NUMBER][MAX_STATES_NUMBER],  /* transition probabilities */
							double symbol_probab[MAX_PARAMETERS_NUMBER][MAX_STATES_NUMBER][MAX_TIME],  /* output symbol probabilities */
							double num_trans_probab[MAX_STATES_NUMBER][MAX_STATES_NUMBER],  /* variable to calculate transition probabilities */
							double *den_trans_probab)  /* variable to calculate transition probabilities */

{
	double 	product;
	double	aux = 0.0;
//	int 	symbol;        /*auxiliar variables */
	register int i,j,k,l;

	for (i=0;i<states_number;i++)
	{
		 for (j=0;j<states_number;j++)
		 {
			  if (j >= i && j < (i+DELTA+1))
			  {
				  aux = 0.0;
				  for (k=0;k<(obs_time-1);k++)
				  {
					   product = 1.0;
					   for (l=0;l<param_number;l++)
					   {
							product *= symbol_probab[l][j][k+1];
					   }
					   aux += alpha[i][k]*transition_probab[i][j]*product*beta[j][k+1];
				  }
				  num_trans_probab[i][j] += aux;
			  }
		 }
		 for (k=0;k<(obs_time-1);k++) den_trans_probab[i] += alpha[i][k]*beta[i][k]/scaling_factor[k];
	}
}   /* end of function calc_transition_probab */






/*
********************************************************************************
*                                                                              *
*   calc_den_mix_coef:                      		                       *
*                                                                              *
*   This function calculates a variable to calculate mixture coefficients.     *
*									       *
*   input: obs_time, states_number, alpha, beta, scaling_factor,               *
*          den_mixture_coef.                                                   *
*									       *
*   output: den_mixture_coef.                                                  *
*									       *
********************************************************************************
*/

void calc_den_mix_coef(int obs_time,           /* time */
					   int states_number,      /* number of states */
					   double alpha[MAX_STATES_NUMBER][MAX_TIME], /* forward variable - alpha */
					   double beta[MAX_STATES_NUMBER][MAX_TIME],   /* backward variable - beta */
					   double *scaling_factor,  /* scaling factor */
					   double den_mixture_coef[MAX_STATES_NUMBER]) /* variable to calculate mixture coefficients */
{
	double 			aux;     /* auxliar variable */
	register int 	i,j;

	for (i=0;i<states_number;i++)
	{
		 for (j=0;j<obs_time;j++)
		 {
			  aux = alpha[i][j]*beta[i][j]/scaling_factor[j];
		 	  den_mixture_coef[i] += aux;
		 }
	}
}   /* end of function calc_den_mix_coef*/










/*
********************************************************************************
*                                                                              *
*   calc_mix_param:                      		                       *
*                                                                              *
*   This function calculates mixture parameters.                               *
*									       *
*   input: obs_time,states_number, mixture_number, coef_number, coef_vector,   *
*          alpha, beta, scaling_factor, gaus_probab_den, num_mix_param,        *
*          state_mix.                                                          *
*                                                                              *
*   output: num_mix_param.                                                     *
*									       *
********************************************************************************
*/

void calc_mix_param(int obs_time,             /* time */
					int states_number,        /* number of states */
					int mixture_number,       /* number of mixtures */
					int coef_number,          /* number of coefficients per vector */
					double *coef_vector,      /* vector of coefficients */
					double alpha[MAX_STATES_NUMBER][MAX_TIME], /* forward variable - alpha */
					double beta[MAX_STATES_NUMBER][MAX_TIME],   /* backward variable - beta */
					double *scaling_factor,   /* scaling factor */
					double gaus_probab_dens[MAX_STATES_NUMBER][MAX_MIXTURE_NUMBER],  /* gaussian probability density */
					struct state *num_mix_param,  /* variable to calculate transition probabilities */
					struct state *state_mix)  /* mixture parameters */

{
	double 			dif[MAX_COEF_NUMBER];         /* auxiliar variable  to calculate  covariance matrix */

	double 			aux;        /* auxiliar variables */
	double	   		aux1;
	register int 	i,j,k,l;

	for (i=0;i<states_number;i++)
	{
		 aux = alpha[i][obs_time]*beta[i][obs_time]/scaling_factor[obs_time];

		 for (j=0;j<mixture_number;j++)
		 {
			   aux1 = aux*gaus_probab_dens[i][j];
			   num_mix_param[i].mix_coef[j] += aux1;

			   for (k=0;k<coef_number;k++)
			   {
					num_mix_param[i].mix[j].mean[k] += aux1*coef_vector[k];
					dif[k] = coef_vector[k] - state_mix[i].mix[j].mean[k];
			   }

			   for (k=0;k<coef_number;k++)
					for (l=k;l<coef_number;l++)
						 num_mix_param[i].mix[j].cov_matrix[k][l] += aux1*dif[k]*dif[l];
		 }
	}
}   /* end of function calc_mix_param */






/*
********************************************************************************
*                                                                              *
*   calc_symbol_probab:                   		                       *
*                                                                              *
*   This function calculates the output symbol density probabilities.          *
*									       *
*   input: states_number, mixture_number,coef_number, coef_vector, state_mix,  *
*          obs_time.                                                           *
*									       *
*   output: gauss, symbol_probab.                                              *
*									       *
********************************************************************************
*/

void calc_symbol_probab(int states_number,    /* number of states */
						int mixture_number,   /* number of mixtures */
						int coef_number,      /* number of coefficients per vector */
						double *coef_vector,  /* vector of coefficients */
						struct state *state_mix,  /* mixture parameters */
						double gauss[MAX_STATES_NUMBER][MAX_MIXTURE_NUMBER],  /* gaussian probability density in each mixture */
						double symbol_probab[MAX_STATES_NUMBER][MAX_TIME],  /* gaussian probability density in each state */
						int obs_time)        /* time */
{
	register int i,j;

	for (i=0;i<states_number;i++)
	{
		 symbol_probab[i][obs_time] = 0.0;
		 for (j=0;j<mixture_number;j++)
		 {
			  gauss[i][j] = calc_gaus(coef_number,coef_vector,
									  state_mix[i].mix[j].mean,state_mix[i].mix[j].cov_matrix,
									  state_mix[i].mix[j].det);
			  gauss[i][j] *= state_mix[i].mix_coef[j];
			  symbol_probab[i][obs_time] += gauss[i][j];
		 }

		 if (symbol_probab[i][obs_time]!= 0.0)
		 {
			 for (j=0;j<mixture_number;j++)
			 {
				 gauss[i][j] /= symbol_probab[i][obs_time];
			 }
		 }
		 else
		 {
			 for (j=0;j<mixture_number;j++)
			 {
				 gauss[i][j] = 0.0;
			 }
		 }
	}
}   /* end of function calc_symbol_probab */





/*
********************************************************************************
*                                                                              *
*   calc_gaus:                           		                       *
*                                                                              *
*   This function calculates the gaussian probability density using full       *
*   covariance matrix.                                                         *
*									       *
*   input: coef_number, coef_vector, mean, inv_cov, det.                       *
*									       *
*   output: gaus.                                                              *
*									       *
********************************************************************************
*/

double calc_gaus(int coef_number,                     /* number of coefficients per vector */
				 double *coef_vector,                 /* vector of coefficients */
				 double *mean,                        /* mixture mean vector */
				 double inv_cov[MAX_COEF_NUMBER][MAX_COEF_NUMBER],     /* inverse covariance matrix */
				 double det)                          /* covariance matrix determinant */
{
	double 		dif[MAX_COEF_NUMBER];
	double 		aux = 0.0;                    /* auxiliar variables */
	double	  	aux1;
	double	   	aux2;
	double	   	temp;
	double	   	gaus;

	register int i,j;

	/* gaussian probability density computation */

	aux1 = 2.0*M_PI;
	aux2 = coef_number/2.0;
	aux1 = pow(aux1,aux2);

	if (det != 0)
	{
		aux2 = fabs(det);
		aux2 = pow(aux2,0.5);

		for (i=0;i<coef_number;i++)
		{
			dif[i] = coef_vector[i] - mean[i];
		}

		for (i=0;i<coef_number;i++)
		{
			 temp = 0.0;
			 for (j=0;j<coef_number;j++)
			 {
				 temp += dif[j]*inv_cov[j][i];
			 }
			 aux += dif[i]*temp;
		}

		aux *= (-0.5);
		aux = exp(aux);

		gaus = aux/(aux1*aux2);

		if (isinf(gaus) == 1)
		{
			gaus = 1e20;
		}
	}

	return(gaus);
}   /* end of function calc_gaus */





/*
********************************************************************************
*                                                                              *
*   updating_transition_probab:              		                       *
*                                                                              *
*   This function updates the transition probabilities.                        *
*									       *
*   input: states_number, num_trans_probab, den_trans_probab.                  *
*									       *
*   output: transition_probab.                                                 *
*									       *
********************************************************************************
*/

void updating_transition_probab(int states_number,         /* number of states */
								double num_trans_probab[MAX_STATES_NUMBER][MAX_STATES_NUMBER],   /* variable to calculate transition probabilities */
								double *den_trans_probab,  /* variable to calculate transition probabilities */
								double transition_probab[MAX_STATES_NUMBER][MAX_STATES_NUMBER])  /* transition probabilities */
{
	double sum;
	register int i,j;

	/* updating transition probabilities */
	for (i=0;i<states_number;i++)
	{
		 if (den_trans_probab[i] != 0.0)
		 {
			 sum = 0.0;
			 for (j=0;j<states_number;j++)
			 {
				  transition_probab[i][j] = num_trans_probab[i][j]/den_trans_probab[i];
				  sum += transition_probab[i][j];
			 }
			 if (sum > 1.001 || sum < 0.999) printf("error on computing transition probabilities: sum = %f \n",sum);
		 }
	}
}    /* end of function updating_transition_probab */





/*
********************************************************************************
*                                                                              *
*   updating_mix_param:                 		                       *
*                                                                              *
*   This function updates the mixture parameters.                              *
*									       *
*   input: states_number, mixture_number, coef_number, den_mixture_coef,       *
*          num_mix_coef, state_mix.                                            *
*									       *
*   output: state_mix.                                                         *
*									       *
********************************************************************************
*/


void updating_mix_param(int states_number,           /* number of states */
						int mixture_number,          /* number of mixtures */
						int coef_number,             /* number of coefficients per vector */
						double *den_mixture_coef,    /* variable to calculate mixture coefficients */
						struct state *num_mix_param, /* variable to calculate transition probabilities */
						struct state *state_mix)    /* mixture parameters */
{
	double sum ;
	register int i,j,k,l;

	for (i=0;i<states_number;i++)
	{
		 if (den_mixture_coef[i] != 0.0)
		 {
			 for (j=0;j<mixture_number;j++)
			 {
				  state_mix[i].mix_coef[j] = num_mix_param[i].mix_coef[j]/den_mixture_coef[i];
				  for (k=0;k<coef_number;k++)
				  {
					   state_mix[i].mix[j].mean[k] = num_mix_param[i].mix[j].mean[k]/num_mix_param[i].mix_coef[j];
					   for (l=k;l<coef_number;l++)
							state_mix[i].mix[j].cov_matrix[k][l] = num_mix_param[i].mix[j].cov_matrix[k][l]/num_mix_param[i].mix_coef[j];
				  }

				  for (k=0;k<coef_number;k++)
					   if (state_mix[i].mix[j].cov_matrix[k][k] < FINITE_PROBAB)
						   state_mix[i].mix[j].cov_matrix[k][k] = FINITE_PROBAB;

				  for (k=1;k<coef_number;k++)
					   for(l=0;l<k;l++)
						   state_mix[i].mix[j].cov_matrix[k][l] =  state_mix[i].mix[j].cov_matrix[l][k];
			 }

		 }

	}

	/* changing values of  less than a threshould  */
	for (i=0;i<states_number;i++)
		 changing_zero_coef(mixture_number,state_mix[i].mix_coef);

	for (i=0;i<states_number;i++)
	{
		 sum = 0.0;
		 for (j=0;j<mixture_number;j++)
			  sum += state_mix[i].mix_coef[j];
		 if (sum > 1.001 || sum < 0.999)
			 printf("error on computing mixture coefficients: sum = %f \n",	sum);
	}
}    /* end of function updating_mixture_param */




/*
********************************************************************************
*                                                                              *
*   calc_det:                             		                       *
*                                                                              *
*   This function computes the diagonal matrix determinant.                    *
*									       *
*   input: coef_number, matrix.                                                *
*									       *
*   output: det.                                                               *
*									       *
********************************************************************************
*/


double calc_det(int coef_number,     /* number of coefficients per vector */
				double *d_matrix)    /* diagonal matrix */
{
	double det;                        /* matrix determinant */
	register int i;

	det = 1.0;

	for (i=0;i<coef_number;i++) det  *= d_matrix[i];

	return(det);

}    /* end of function calc_det */








/*
********************************************************************************
*                                                                              *
*   decomposition:                             		                       *
*                                                                              *
*   This function decomposes the covariance matrix as T' D T,                  *
*   where: D is a diagonal matrix, T is a lower triangular matrix (whose main  *
*   diagonal elements are all 1's).                                            *
*									       *
*   input: coef_number, cov_matrix.                                            *
*									       *
*   output: d_matrix, t_matrix.                                                *
*									       *
********************************************************************************
*/


void decomposition(int coef_number,                     /* number of coefficients per vector */
				   double cov_matrix[MAX_COEF_NUMBER][MAX_COEF_NUMBER],  /* covariance matrix */
				   double *d_matrix,                    /* diagonal matrix */
				   double t_matrix[MAX_COEF_NUMBER][MAX_COEF_NUMBER])  /* triangular matrix */
{
	register int i,j,k;

	for (i=0;i<coef_number;i++)	 d_matrix[i] = 0.0;

	for (i=0;i<coef_number-1;i++)
	{
		 t_matrix[i][i] = 1.0;
		 for (j=i+1;j<coef_number;j++)	t_matrix[i][j] = 0.0;
	}

	t_matrix[coef_number-1][coef_number-1] = 1.0;

	j = 0;
	d_matrix[0] = cov_matrix[0][0];

	for (i=1;i<coef_number;i++)	 t_matrix[i][0] = cov_matrix[i][0]/d_matrix[0];

	for (j=1;j<coef_number-1;j++)
	{
		  d_matrix[j] = cov_matrix[j][j];
		  for (k=0;k<j;k++)  d_matrix[j] -= t_matrix[j][k]*t_matrix[j][k]*d_matrix[k];
		  for (i=j+1;i<coef_number;i++)
		  {
			   t_matrix[i][j] = cov_matrix[i][j];
			   for (k=0;k<j;k++)	t_matrix[i][j] -= t_matrix[i][k]*d_matrix[k]*t_matrix[j][k];
			   t_matrix[i][j] /= d_matrix[j];
		  }
	}

	j= coef_number-1;
	d_matrix[j] = cov_matrix[j][j];
	for (k=0;k<j;k++)	d_matrix[j] -= t_matrix[j][k]*t_matrix[j][k]*d_matrix[k];

}    /* end of function decomposition */






/*
********************************************************************************
*                                                                              *
*   inv_triang_matrix:                                                         *
*                                                                              *
*   This function inverts a lower triangular matrix.                           *
*									       *
*   input: coef_number, matrix.                                                *
*									       *
*   output: i_matrix.                                                          *
*									       *
********************************************************************************
*/


void inv_triang_matrix(int coef_number,                     /* number of coefficients per vector */
					   double matrix[MAX_COEF_NUMBER][MAX_COEF_NUMBER], /* triangular matrix */
					   double i_matrix[MAX_COEF_NUMBER][MAX_COEF_NUMBER])  /* inverse triangular matrix */
{
	register int i,j,k,l;

	for (i=0;i<coef_number-1;i++)
	{
		 i_matrix[i][i] = 1.0;
		 for (j=i+1;j<coef_number;j++)
		 i_matrix[i][j] = 0.0;
	}

	i_matrix[coef_number-1][coef_number-1] = 1.0;

	for (k=0;k<coef_number-1;k++)
	{
		 for (i=k+1;i<coef_number;i++)
		 {
			  j = i-k-1;
			  i_matrix[i][j] = 0.0;
			  for (l=j;l<i;l++)	i_matrix[i][j] -= matrix[i][l]*i_matrix[l][j];
		 }
	}
}    /* end of function inv_triang_matrix */






/*
********************************************************************************
*                                                                              *
*   inv_cov_matrix:                                                            *
*                                                                              *
*   This function inverts a covariance matrix.                                 *
*									       *
*   input: coef_number, cov_matrix.                                            *
*									       *
*   output: cov_matrix.                                                        *
*									       *
********************************************************************************
*/


double inv_cov_matrix(int coef_number,                     /* number of coefficients per vector */
					  double cov_matrix[MAX_COEF_NUMBER][MAX_COEF_NUMBER])  /* covariance matrix */
{
	double d_matrix[MAX_COEF_NUMBER];                   /* diagonal matrix */
	double t_matrix[MAX_COEF_NUMBER][MAX_COEF_NUMBER];  /* triangular matrix */
	double i_matrix[MAX_COEF_NUMBER][MAX_COEF_NUMBER];  /* inverse triangular matrix */
	double det;
	register int i,j,k;

	decomposition(coef_number,cov_matrix,d_matrix,t_matrix);
	det = calc_det(coef_number,d_matrix);

	if (isnan(det) == 1)	det = 0.0;


	if (det != 0.0)
	{
		inv_triang_matrix(coef_number,t_matrix,i_matrix);

		for (i=0;i<coef_number;i++)
		{
			 cov_matrix[i][i] = 0.0;
			 for (j=i;j<coef_number;j++)
				  cov_matrix[i][i] += i_matrix[j][i]*i_matrix[j][i]/d_matrix[j];
		}

		for (i=0;i<coef_number-1;i++)
		{
			 for (j=i+1;j<coef_number;j++)
			 {
				  cov_matrix[i][j] = 0.0;
				  for (k=j;k<coef_number;k++)
					   cov_matrix[i][j] += i_matrix[k][i]*i_matrix[k][j]/d_matrix[k];
				  cov_matrix[j][i] = cov_matrix[i][j];
			 }
		}
	}
	return(det);
}    /* end of function inv_cov_matrix */







/*
********************************************************************************
*                                                                              *
*   treat_zero_det:        				                       *
*                                                                              *
*   This function creates a new parameters for a mixture with covariance       *
*   matrix equal to zero by splitting the parameters of a mixture with the     *
*   biggest coefficient.                                                       *
*									       *
*   input: coef_number, mixture_number, state_mix.        		       *
*									       *
*   output: state_mix.                                                        *
*									       *
********************************************************************************
*/
 
void treat_zero_det(int coef_number,                     /* number of coefficients */
					int mixture_number,                  /* number of mixtures */
					struct state *state_mix)             /* pointer to mixtures */
{
	double vector[MAX_MIXTURE_NUMBER];
	int index_sort[MAX_MIXTURE_NUMBER];
	double sum;
	register int i,j,k,l,m;

	for (j=0;j<mixture_number;j++)	vector[j] = state_mix->mix[j].det;

	sorting(vector,index_sort,mixture_number);

	i = 0;
	for (j=0;j<mixture_number;j++)
	{
		 if (state_mix->mix[j].det < 1e-20)
		 {
			 l = index_sort[i++];
			 for (k=0;k<coef_number;k++)
				  state_mix->mix[j].mean[k] = state_mix->mix[l].mean[k]*(1.05);
			 for (k=0;k<coef_number;k++)
				  state_mix->mix[l].mean[k] = state_mix->mix[l].mean[k]*(0.95);
			 for (m=0;m<coef_number;m++)
				  for (k=0;k<coef_number;k++)
					   state_mix->mix[j].cov_matrix[m][k] =  state_mix->mix[l].cov_matrix[m][k];

			 state_mix->mix[j].det = state_mix->mix[l].det;

			 state_mix->mix_coef[l] /= 2.0;
			 state_mix->mix_coef[j] = state_mix->mix_coef[l];
		 }   /* end of if */
	}  /* end of for */

	sum = 0.0;
	for (j=0;j<mixture_number;j++)	sum += state_mix->mix_coef[j];

	for (j=0;j<mixture_number;j++)	state_mix->mix_coef[j] /= sum;

}  /* end of function treat_zero_det */





/*
********************************************************************************
*                                                                              *
*   writing_model:      				                       *
*                                                                              *
*   This function writes the model created.                                    *
*									       *
*   input: output_file, states_number, param_number, mixture_number,           *
*          coef_number,transition_probab, state_mix, word.                     *
*									       *
*   output: void.                                                              *
*									       *
********************************************************************************
*/

void writing_model(char *output_file,         /* output file name */
				   int states_number,         /* number of states */
				   int param_number,          /* number of parameters */
				   int *mixture_number,       /* number of mixtures */
				   int *coef_number,          /* number of coefficients */
				   double transition_probab[MAX_STATES_NUMBER][MAX_STATES_NUMBER],  /* transition probabilities */
				   struct state state_mix[MAX_PARAMETERS_NUMBER][MAX_STATES_NUMBER],  /* mixtures parameters */
				   char *word)  /* word that is represented by the model */
{
	register int i,j,k,l;
	size_t length;

	length = strlen(word);

	fwrite(&length,sizeof(size_t),1,f_out); /* writing length of word */
	if (ferror(f_out))
	{
		printf("writing error on file %s \n",output_file);
		exit(1);
	}

	fwrite(word,sizeof(char),length,f_out); /* writing word */
	if (ferror(f_out))
	{
		printf("writing error on file %s \n",output_file);
		exit(1);
	}

	fwrite(&states_number, sizeof(int),1,f_out);   /* writing number of states */
	if (ferror(f_out))
	{
		printf("writing error on file %s \n",output_file);
		exit(1);
	}

	fwrite(&param_number, sizeof(int),1,f_out);   /* writing number of parameters */
	if (ferror(f_out))
	{
		printf("writing error on file %s \n",output_file);
		exit(1);
	}

	fwrite(mixture_number,sizeof(int),param_number,f_out);   /* writing number of mixtures */
	if (ferror(f_out))
	{
		printf("writing error on file %s \n",output_file);
		exit(1);
	}

	fwrite(coef_number,sizeof(int),param_number,f_out);   /* writing number of coefficients */
	if (ferror(f_out))
	{
		printf("writing error on file %s \n",output_file);
		exit(1);
	}


	/* writing transition probabilities */
	for (i=0;i<states_number;i++)
	{
		 fwrite(transition_probab[i],sizeof(double),states_number,f_out);
		 if (ferror(f_out))
		 {
			 printf("writing error on file %s \n",output_file);
			 exit(1);
		 }
	}

	/* writing mixture parameters */
	for (i=0;i<param_number;i++)
	{
		 for (j=0;j<states_number;j++)
		 {

			  /* writing mixture coefficients */
			  fwrite(state_mix[i][j].mix_coef,sizeof(double),mixture_number[i],f_out);
			  if (ferror(f_out))
			  {
				  printf("writing error on file %s \n",output_file);
				  exit(1);
			   }

			  for (k=0;k<mixture_number[i];k++)
			  {
				   /* writing mixture mean */
				   fwrite(state_mix[i][j].mix[k].mean,sizeof(double),coef_number[i],f_out);
				   if (ferror(f_out))
				   {
					   printf("writing error on file %s \n",output_file);
					   exit(1);
				   }

				   /* writing covariance matrix determinat */
				   fwrite(&(state_mix[i][j].mix[k].det),sizeof(double),1,f_out);
				   if (ferror(f_out))
				   {
					   printf("writing error on file %s \n",output_file);
					   exit(1);
				   }

				   /* writing  inverse covariance matrix (full) */
				   for (l=0;l<coef_number[i];l++)
				   {
						fwrite(state_mix[i][j].mix[k].cov_matrix[l],sizeof(double),coef_number[i],f_out);
						if (ferror(f_out))
						{
							printf("writing error on file %s \n",output_file);
							exit(1);
						}
				   }
			  }
		 }
	}
}    /* end of function writing_model */




/*
********************************************************************************
*                                                                              *
*   writing_text:       				                       *
*                                                                              *
*   This function writes a text file.                                          *
*									       *
*   input: out_file2, output_file,word, states_number, param_number,           *
*          mixture_number,data_file,starting_time_f,ending_time_f,cpu_time_f,  *
*          exemplar_number,probab,iteration.                                   *
*									       *
*   output: void.                                                              *
*									       *
********************************************************************************
*/


void writing_text(char *out_file2,        /* text file name */
				  char *output_file,      /* model file name */
				  char *word,             /* word that is represented by the model */
				  int states_number,      /* number of states */
				  int param_number,       /* number of parameters */
				  int *mixture_number,    /* number of mixtures */
				  char data_file[MAX_PARAMETERS_NUMBER][100], /* data file */
				  char *starting_time_f,  /* starting time */
				  char *ending_time_f,    /* ending time */
				  char *cpu_time_f,       /* cpu time */
				  int exemplar_number,    /* number of exemplar in training sequence */
				  double probab,          /* mean probab */
				  int iteration)          /* number of iterations */
{
	register int i;

	fprintf(f_out,"Continuous HMM created using Forward Backward algorithm. It is considered full covariance matrix. It is considered a final state.\n");
	if (ferror(f_out))
	{
		printf("writing error on file %s \n",out_file2);
		exit(1);
	}
	fprintf(f_out,"model file: %s \n", output_file);
	if (ferror(f_out))
	{
		printf("writing error, file %s \n",out_file2);
		exit(1);
	}
	fprintf(f_out,"word: %s \n", word);
	if (ferror(f_out))
	{
		printf("writing error, file %s \n",out_file2);
		exit(1);
	}
	fprintf(f_out,"number of states: %d \n", states_number);
	if (ferror(f_out))
	{
		printf("writing error on  file %s \n",out_file2);
		exit(1);
	}
	fprintf(f_out,"number of parameters: %d \n", param_number);
	if (ferror(f_out))
	{
		printf("writing error on file %s \n",out_file2);
		exit(1);
	}
	for (i=0;i<param_number;i++)
	{
		 fprintf(f_out,"number of mixtures %d: %d \n", (i+1),mixture_number[i]);
		 if (ferror(f_out))
		 {
			 printf("writing error on file %s \n",out_file2);
			 exit(1);
		 }
	}
	for (i=0;i<param_number;i++)
	{
		 fprintf(f_out,"parameter %d: %s \n", (i+1),data_file[i]);
		 if (ferror(f_out))
		 {
			 printf("writing error on file %s \n",out_file2);
			exit(1);
		 }
	}
	fprintf(f_out,"threshould to finish training: %f \n",THRESHOULD);
	if (ferror(f_out))
	{
		printf("writing error on file %s \n",out_file2);
		exit(1);
	}
	fprintf(f_out,"number of exemplars in training sequence: %d \n",exemplar_number);
	if (ferror(f_out))
	{
		printf("writing error on file %s \n",out_file2);
		exit(1);
	}
	fprintf(f_out,"mean probability: %f \n",probab);
	if (ferror(f_out))
	{
		printf("writing error on file %s \n",out_file2);
		exit(1);
	}
	fprintf(f_out,"number of iterations: %d \n",iteration);
	if (ferror(f_out))
	{
		printf("writing error on file %s \n",out_file2);
		exit(1);
	}
	fprintf(f_out,"starting time: %s \n",starting_time_f);
	if (ferror(f_out))
	{
		printf("writing error on file %s \n",out_file2);
		exit(1);
	}
	fprintf(f_out,"ending time: %s \n",ending_time_f);
	if (ferror(f_out))
	{
		printf("writing error on file %s \n",out_file2);
		exit(1);
	}
	fprintf(f_out,"cpu time: %s \n",cpu_time_f);
	if (ferror(f_out))
	{
		printf("writing error on file %s \n",out_file2);
		exit(1);
	}
}   /* end of function writing_text */





