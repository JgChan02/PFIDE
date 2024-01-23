#include "Self_Define_Functions.h"
#include <time.h>
#include <cstdio>
//#include <windows.h>
#include <algorithm>
#include <iostream>
#include <fstream>
#include <stdlib.h>
#include <iomanip>
#include <cstring>
#include <string>
#include <boost/random.hpp>
#include <boost/random/uniform_int.hpp>
#include <boost/random/cauchy_distribution.hpp>
#include <boost/random/normal_distribution.hpp>
#include <boost/random/uniform_real.hpp>
#include <vector>
#include <numeric>

using namespace std;
#include <stdio.h>
#include <math.h>
#include <malloc.h>
#define PI 3.1415926535897932384626433832795029


// in this program the global best solution is updated after each subgroup finish iterating itself
int main(int argc, char* argv[])
{
    int i, j, k;
    int Population_size;
for (Population_size = 100; Population_size <= 230; Population_size += 10)
{
    //////////////////////////global processing/////////////////////////////////////
    int funToRun[] = { 1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30 };  //function set

    int funNum = 30; //total function num
    int best_fitness_value[30] = { 100,200,300,400,500,600,700,800,900,1000,1100,1200,1300,1400,1500,1600,1700,1800,1900,2000,2100,2200,2300,2400,2500,2600,2700,2800,2900,3000 };

    int function_index;
    int run_index;

    double* Lbound = new double[dim];
    double* Ubound = new double[dim];

    double all_results[timesOfRun][record_set_size];
    ////////////////////////////////////////////////////////////////////////////////

    //////////////////////////JADE processing//////////////////////////////////////
    //种群数量与记录档案数量相同，代码中均使用Population_size这个数值
    int parent1, parent2, Rank_num, r;

    double q = 0.3;
    double p;
    int num_of_feature = 4;

    double global_fitness;
    double trail_vector_result;
    double average_of_fitness;
    double sum_of_delta_fitness;
    double sum_of_w_F_squared;
    double sum_of_w_F;
    double sum_of_w_CR;

    vector<double> good_CR;
    vector<double> good_F;
    vector<double> delta_fitness;
    vector<double> w;

    double* average_of_dim = new double [dim];
    double* global_best = new double[dim];
    double* mutation_vector = new double[dim];
    double* trail_vector = new double[dim];
    double* results = new double[Population_size];// the fitness results for the whole population
    double* std_of_dim = new double [dim];
    double* F_individual = new double [Population_size];
    double* CR_individual = new double [Population_size];
    double* population_feature_similarity = new double [Population_size];

    double** population = new double* [Population_size];
    double** population_feature = new double *[Population_size];
    for (i = 0; i < Population_size; ++i)
    {
        population[i] = new double[dim];
        population_feature[i] = new double [num_of_feature];
    }


    double** A = new double* [Population_size];
    for (i = 0; i < Population_size; ++i)
        A[i] = new double[dim];

    NewType* temp = new NewType[Population_size];
    int* ranking_index = new int[Population_size];

    int FV;
    int fv_counter;
    int A_num;
    int index_del;

    int position;
    double std_of_fitness;
    double std_of_population;

    double MAX = 100, MIN = -100;
    //////////////////////////////////////////////////////////////////////////////

    for (function_index = 0; function_index < funNum; function_index++)
    {
        cout << "Function " << funToRun[function_index] << " Begined!" << endl;

        boost::mt19937 generator(time(0) * rand());

        // to initialize the population
        boost::uniform_real<> uniform_real_generate_x(MIN, MAX);
        boost::variate_generator< boost::mt19937&, boost::uniform_real<> > random_real_num_x(generator, uniform_real_generate_x);

        boost::uniform_real<> uniform_real_generate_r(0, 1);
        boost::variate_generator< boost::mt19937&, boost::uniform_real<> > random_real_num_r(generator, uniform_real_generate_r);

        //select an individual from the population
        boost::uniform_int<> int_generator1(0, Population_size - 1);
        boost::variate_generator< boost::mt19937&, boost::uniform_int<> > int_number1(generator, int_generator1);

        boost::uniform_int<> int_generator_rank1(0, (int)(q * Population_size) - 1);
        boost::variate_generator< boost::mt19937&, boost::uniform_int<> > int_number_rank1(generator, int_generator_rank1);

        boost::uniform_real<> uniform_real_generate_r1(2.0/Population_size, 0.2);
        boost::variate_generator< boost::mt19937&, boost::uniform_real<> > random_real_num_r1(generator, uniform_real_generate_r1);


        //select an individual from the top best ones

        for (i = 0; i < dim; ++i)
        {
            Lbound[i] = MIN;
            Ubound[i] = MAX;
        }

        char fun1[10];
	char fun2[10];
	char fun3[10];
	snprintf(fun1, 10, "%d", funToRun[function_index]);
	snprintf(fun2, 10, "%d", dim);
	snprintf(fun3, 10, "%d", Population_size);
        string filename_fitness = "./Results/"+ to_string(Population_size) +"/F" + to_string(funToRun[function_index]) + ".txt";
        ofstream out_fitness(filename_fitness.c_str());

        if (!out_fitness)
        {
            cerr << "Can not open the file " << filename_fitness << endl;
            exit(1);
        }


        for (run_index = 0; run_index < timesOfRun; run_index++)
        {
            FV = 0;
            fv_counter = 0;
            position = 0;
            A_num = 0;

            cout << "Running the " << run_index << "th times" << endl;

            //initialize the population
            for (i = 0; i < Population_size; ++i)
            {
                for (j = 0; j < dim; ++j)
                {
                    population[i][j] = random_real_num_x();
                }

                for(k=0;k<num_of_feature;k++)
                {
                    if(k<=1)
                    {
                        population_feature[i][k]= numeric_limits<double>::infinity();
                    }
                    else
                    {
                        population_feature[i][k]=0.5;
                    }
                }
            }

            for (i = 0; i < Population_size; i++)
            {
                cec17_test_func(population[i], &results[i], dim, 1, function_index + 1);
                results[i] = results[i] - best_fitness_value[function_index];
            }

            FV += Population_size;

            Find_best(results, Population_size, global_best, population, global_fitness, dim);


            while (FV < MAX_FV)
            {
                average_of_fitness = 0;
                std_of_fitness = 0;
                std_of_population = 0;

                if (fv_counter < record_set_size && FV >= MAX_FV)
                {
                    all_results[run_index][fv_counter] = global_fitness;
                    fv_counter++;
                }

                for(i=0;i<dim;i++)
                {
                    average_of_dim[i] = 0;
                    std_of_dim[i] = 0;
                }

                for(i=0;i<Population_size;i++)
                {
                    average_of_fitness += results[i];
                }
                average_of_fitness /= Population_size;

                for(j=0;j<dim;j++)
                {
                    for(i=0;i<Population_size;i++)
                    {
                        average_of_dim[j] += population[i][j];
                    }
                    average_of_dim[j] /= Population_size;
                }

                for(i=0;i<Population_size;i++)
                {
                    std_of_fitness += pow(results[i]-average_of_fitness,2);
                }
                std_of_fitness = sqrt(std_of_fitness);

               for(j=0;j<dim;j++)
               {
                   for(i=0;i<Population_size;i++)
                   {
                       std_of_dim[j] += pow(population[i][j]-average_of_dim[j],2);
                   }
                   std_of_dim[j] = sqrt(std_of_dim[j]);
               }

               for(j=0;j<dim;j++)
               {
                   std_of_population += std_of_dim[j];
               }

               for(i=0;i<Population_size;i++)
               {
                   population_feature_similarity[i] = sqrt(pow(std_of_fitness-population_feature[i][0],2)+pow(std_of_population-population_feature[i][1],2));
               }
               Ranking(ranking_index, population_feature_similarity, temp, Population_size);


                for(i=0;i<Population_size;i++)
                {
                   r=int_number_rank1();
                   boost::normal_distribution<> CR_generator(population_feature[ranking_index[r]][3], 0.1);
                   boost::variate_generator< boost::mt19937&, boost::normal_distribution<> > CR_number(generator, CR_generator);
                   boost::cauchy_distribution<> F_generator(population_feature[ranking_index[r]][2], 0.1);
                   boost::variate_generator< boost::mt19937&, boost::cauchy_distribution<> > F_number(generator, F_generator);

                   F_individual[i] = F_number();
                   while (F_individual[i] <= 0 || F_individual[i] >= 1)
                   {
                       F_individual[i] = F_number();
                   }

                   CR_individual[i] = CR_number();
                   while (CR_individual[i] >= 1 || CR_individual[i] <= 0)
                   {
                       CR_individual[i] = CR_number();
                   }
               }

                Ranking(ranking_index, results, temp, Population_size);

                p = random_real_num_r1();

                boost::uniform_int<> int_generator2(0, Population_size + A_num - 1);
                boost::variate_generator< boost::mt19937&, boost::uniform_int<> > int_number2(generator, int_generator2);
                boost::uniform_int<> int_generator_rank(0, (int)(p * Population_size) - 1);
                boost::variate_generator< boost::mt19937&, boost::uniform_int<> > int_number_rank(generator, int_generator_rank);

                for (i = 0; i < Population_size; ++i)
                {
                    Rank_num = int_number_rank();
                    Rank_num = ranking_index[Rank_num];
                    while (Rank_num == i)
                    {
                        Rank_num = ranking_index[int_number_rank()];
                    }

                    parent1 = int_number1();
                    while (parent1 == i||parent1 ==Rank_num)
                    {
                        parent1 = int_number1();
                    }

                    parent2 = int_number2();
                    while (parent2 == i || parent2 == parent1||parent2 ==Rank_num)
                    {
                        parent2 = int_number2();
                    }

                    if (parent2 <= Population_size - 1)
                    {
                        Mutation(mutation_vector, population[i], population[parent1], population[parent2], population[Rank_num], Lbound, Ubound, F_individual[i], dim);
                    }
                    else
                    {
                        parent2 = parent2 - Population_size;
                        Mutation(mutation_vector, population[i], population[parent1], A[parent2], population[Rank_num], Lbound, Ubound, F_individual[i], dim);
                    }

                    Crossover(trail_vector, population[i], mutation_vector, CR_individual[i], dim);

                    cec17_test_func(trail_vector, &trail_vector_result, dim, 1, function_index + 1);
                    trail_vector_result = trail_vector_result - best_fitness_value[function_index];

                    FV++;

                    if (trail_vector_result < results[i])
                    {
                        delta_fitness.push_back(abs(trail_vector_result-results[i]));

                        results[i] = trail_vector_result;
                        if( A_num > Population_size)
                        {
                            index_del = int_number1();
                            memcpy(A[index_del], population[i], sizeof(double) * dim);//记录劣质解

                        }
                        else
                        {
                            memcpy(A[A_num], population[i], sizeof(double) * dim);//记录劣质解
                            A_num++;
                        }

                        memcpy(population[i], trail_vector, sizeof(double) * dim);


                        good_CR.push_back(CR_individual[i]);
                        good_F.push_back(F_individual[i]);

                        if (results[i] < global_fitness)
                        {
                            global_fitness = results[i];
                        }
                    }

                    if (fv_counter < record_set_size && FV >= MAX_FV)
                    {
                        all_results[run_index][fv_counter] = global_fitness;
                        fv_counter++;
                    }
                }

                if(!good_F.empty() && !good_CR.empty())
                {
                    sum_of_delta_fitness = accumulate(delta_fitness.begin(),delta_fitness.end(),0.0);
                    sum_of_w_CR=0;
                    sum_of_w_F_squared=0;
                    sum_of_w_F=0;
                    int dsize=delta_fitness.size();

                    for(i=0;i<dsize;i++)
                    {
                        w.push_back(delta_fitness[i]/sum_of_delta_fitness);
                    }

                    for(i=0;i<dsize;i++)
                    {
                        sum_of_w_F_squared += w[i]*pow(good_F[i],2);
                        sum_of_w_F += w[i]*good_F[i];
                        sum_of_w_CR += w[i]*good_CR[i];
                    }

                     population_feature[position][3]= sum_of_w_CR;
                     population_feature[position][2]= sum_of_w_F_squared/sum_of_w_F;
                     population_feature[position][0]=std_of_fitness;
                     population_feature[position][1]=std_of_population;
                     position++;
                     if(position > Population_size-1)
                     {
                         position=0;
                     }
                }


                good_F.clear();
                good_CR.clear();
                w.clear();
                delta_fitness.clear();
            }
            out_fitness << global_fitness << endl;
        }
        out_fitness.close();
        cout << "Function " << funToRun[function_index] << " Finished!" << endl;
    }


    //release the resouces
    for (i = 0; i < Population_size; ++i)
    {
        delete[]population[i];
        delete[]population_feature[i];
    }
    delete[]population_feature;
    delete[]population;

    for (i = 0; i < timesOfRun; ++i)
        delete[]all_results[i];
    delete[]all_results;

    for (i = 0; i < Population_size; ++i)
        delete[]A[i];
    delete[]A;

    delete[]temp;
    delete[]ranking_index;
    delete[]results;
    delete[]Lbound;
    delete[]Ubound;
    delete[]trail_vector;
    delete[]mutation_vector;
    delete[]global_best;
    delete[]CR_individual;
    delete[]F_individual;
    delete[]average_of_dim;
    delete[]std_of_dim;
    delete[]population_feature_similarity;
}
    return 0;
}
itness;
                     population_feature[position][1]=std_of_population;
                     position++;
                     if(position > Population_size-1)
                     {
                         position=0;
                     }
                }


                good_F.clear();
                good_CR.clear();
                w.clear();
                delta_fitness.clear();
		        
            }
            out_fitness.close();
            out_diversity.close();
        }
        
        cout << "Function " << funToRun[function_index] << " Finished!" << endl;
    }


    //release the resouces
    for (i = 0; i < Population_size; ++i)
    {
        delete[]population[i];
        delete[]population_feature[i];
    }
    delete[]population_feature;
    delete[]population;

    for (i = 0; i < timesOfRun; ++i)
        delete[]all_results[i];
    delete[]all_results;

    for (i = 0; i < Population_size; ++i)
        delete[]A[i];
    delete[]A;

    delete[]temp;
    delete[]ranking_index;
    delete[]results;
    delete[]Lbound;
    delete[]Ubound;
    delete[]trail_vector;
    delete[]mutation_vector;
    delete[]global_best;
    delete[]CR_individual;
    delete[]F_individual;
    delete[]average_of_dim;
    delete[]std_of_dim;
    delete[]population_feature_similarity;
    delete[]my_col_mean;
}
    return 0;
}
