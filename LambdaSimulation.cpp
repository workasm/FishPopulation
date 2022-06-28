
#include <fstream>
#include <sstream>
#include <iostream>
#include <math.h>
#include "omp.h"

struct Simulator {

struct PointLog2d {
    float x, y, logx;
};

std::vector< PointLog2d > readCsv(const std::string& path) {

    std::ifstream ifs(path);

    if(!ifs.is_open()) {
        throw std::runtime_error("Unable to open the file: " + path);
    }

    std::string line;
    std::getline(ifs, line); // skip caption

    std::vector< PointLog2d > vec;
    vec.reserve(100);
    PointLog2d pt;
    int num; char c;
    while(std::getline(ifs, line)) {

        std::istringstream is(line);
        is >> num >> c >> pt.x >> c >> pt.y;
        vec.push_back(pt);
    }
    return vec;
}

struct FishAgesNum {
    int64_t Nout0;
    int64_t Nout1;
    int64_t Nout2;
};

using Real = double;

void runLambdaInternal(std::ofstream& ofs, const FishAgesNum& opts, Real lambda0, bool isGroupI) {

    struct Params{
        Real beta0 = 0, lambda0 = 0, Mw0 = 0;
        Real beta1 = 0, Mw1 = 0;
        Real sum = std::numeric_limits< Real >::max();
    };

    constexpr int nmins = 500;
    std::vector<Params> g_mins;

    omp_set_num_threads(8);

    constexpr Real step = 0.003;
    constexpr Real lambdaMin = 0.02, lambdaMax = 0.98;
    constexpr Real MpMin = 0.02, MpMax = 0.98;
    constexpr Real MwMin = 0.02, MwMax = 0.98;

    constexpr Real MwFactor = 5;
    constexpr Real betaMinDiff = 0.02, lambdaMinDiff = 0.02;

    const char *group = isGroupI ? "I" : "II";
    fprintf(stderr, "\n--------- simulating group %s for lambda0 = %f ---------\n",
            group, lambda0);

    #pragma omp parallel shared(g_mins)
    {

    int num = (int)((1.0 - step*2) / step);
    Params cur, mins[nmins];
    cur.lambda0 = lambda0;

    #pragma omp for
    for(int i = 0; i < num; i++) {
    //for(cur.beta0 = step; cur.beta0 <= mod - step; cur.beta0 += step, num--) {
    cur.beta0 = step + step*i;

    if(i % 10 == 0)
        fprintf(stderr, "thid: %d; beta0: %.3f -- num: %d\n", omp_get_thread_num(), cur.beta0, i);

    //for(cur.lambda0 = 0.24; cur.lambda0 <= 0.24; cur.lambda0 += step)
    {
        Real Mp = 1.0 - cur.beta0 - cur.lambda0;
        if(!(Mp >= MpMin && Mp <= MpMax))
            continue;

    for(cur.Mw0 = MwMin; cur.Mw0 <= MwMax; cur.Mw0 += step)
    {
        for(cur.Mw1 = MwMin; cur.Mw1 <= MwMax; cur.Mw1 += step) {

        if(!(cur.Mw0 >= cur.Mw1*MwFactor))
            continue;

        for(cur.beta1 = step; cur.beta1 <= 1.0 - step; cur.beta1 += step) {

            Real lambda1 = 1.0 - cur.beta1;
            if(!(cur.lambda0 + lambdaMinDiff < lambda1 &&
                 lambda1 >= lambdaMin && lambda1 <= lambdaMax))
                continue;

            if(!(cur.beta0 > cur.beta1 + betaMinDiff))
                continue;

            auto beta0 = cur.beta0*(1.0 - cur.Mw0), beta1 = cur.beta1*(1.0 - cur.Mw1);
            auto val1 = opts.Nout0*beta0*lambda1 - opts.Nout1*cur.lambda0,
                    val2 = opts.Nout1*beta1 - opts.Nout2*lambda1;

            cur.sum = std::abs((val1)) + std::abs((val2));

            for(int i = 0; i < nmins; i++) {
                if(mins[i].sum > cur.sum) {
//                    memmove(mins + i + 1, mins + i, (nmins-1 - i)*sizeof(Params));
                    for(int j = nmins-1; j > i; j--) {
                        mins[j] = mins[j-1];
                    }
                    mins[i] = cur;
                    break;
                }
            }
        }
    }
    }
    }
    } // for beta0

    #pragma omp critical
    {
        std::vector< Params > zmins(nmins + g_mins.size());
        std::merge(mins, mins + nmins, g_mins.begin(), g_mins.end(), zmins.begin(), [](const Params& p1, const Params& p2)
        {
            return p1.sum < p2.sum;
        });
        g_mins.swap(zmins); // keep only the smallest ones..
    }

    } // parallel clause

    int i = 1; (void)i;
    std::sort(g_mins.begin(), g_mins.end(), [](const Params& p1, const Params& p2){
        return p1.beta0*(1.0 - p1.Mw0) < p2.beta0*(1.0 - p2.Mw0);
    });

    struct Range {
        Real rmin = std::numeric_limits< Real >::max();
        Real rmax = std::numeric_limits< Real >::min();
        Real ravg = 0;
        int total = 0;
    };

    auto setMM = [](Range& r, Real val){
        if(r.rmin > val)
            r.rmin = val;
        if(r.rmax < val)
            r.rmax = val;
        r.ravg += val;
        r.total++;
    };

    Range beta0R, beta1R, Mw0R, Mw1R, lambda0R, lambda1R,
            val1R, val2R;

    //std::stringstream out;

    char bbf[256];
    for(const Params& item: g_mins) {
        Real lambda1 = 1.0 - item.beta1;

        auto beta0 = item.beta0*(1.0 - item.Mw0),
             beta1 = item.beta1*(1.0 - item.Mw1);
        auto val1 = std::abs(opts.Nout0*beta0/item.lambda0 - opts.Nout1/lambda1),
             val2 = std::abs(opts.Nout1*beta1/lambda1 - opts.Nout2);

        auto Mw0 = item.Mw0*item.beta0,
             Mw1 = item.Mw1*item.beta1;

        auto MpMw0 = 1.0 - beta0 - lambda0;

        if(beta0 == 0.0)
            continue;

        if(val1 > 600 || val2 > 600)
            continue;

        setMM(beta0R, beta0);
        setMM(beta1R, beta1);
        setMM(Mw0R, Mw0);
        setMM(Mw1R, Mw1);
        setMM(lambda0R, item.lambda0);
        setMM(lambda1R, lambda1);
        setMM(val1R, val1);
        setMM(val2R, val2);

        sprintf(bbf, "%d ; %.5f ; %.5f ; %.5f / %.5f ; %.5f ; %.5f ; %.5f ; %.2f ; %.2f\n",
            i++, beta0, item.lambda0, MpMw0, Mw0, beta1, lambda1, Mw1, std::abs(val1), std::abs(val2));
        ofs << bbf;
    }
}

void runLambda() {

    //constexpr int64_t shift = 26, mod = 1<<shift;
    // this is a group I
    FishAgesNum groupI = {
        48225, 6746, 2624
    };
    FishAgesNum groupII = {
        9241, 2836, 1103
    };

    constexpr int64_t numFishMin = 475, numFishMax = 475, fishStep = 50;
    constexpr Real numFishEggs = 1100*0.95;

    const Real factorI = 0.85,
               factorII = 1.0 - factorI;

//    std::ofstream ofs("C:\\work\\LambdaSimulation.csv");
//    if(!ofs.is_open()) {
//        std::cerr << "Unable to open file for writing!\n";
//        return;
//    }

//    ofs << "GroupI; Nout0=" << groupI.Nout0 << "; Nout1=" << groupI.Nout1 << "; Nout2=" << groupI.Nout2 << "\n";
//    ofs << "GroupII; Nout0=" << groupII.Nout0 << "; Nout1=" << groupII.Nout1 << "; Nout2=" << groupII.Nout2 << "\n";
//    ofs << "FactorI=" << factorI << "; FactorII=" << factorII << "\n";

    //ofs << "No ; beta0*(1-Mw0) ; lambda0 ; Mp; Mw0 ; beta1*(1-Mw1) ; lambda1 ; Mw1 ; |A1-A2| ; |B1-B2|\n";
    //ofs << "No ; beta0*(1-Mw0) ; lambda0 ; Mp; beta0*Mw0 ; beta1*(1-Mw1) ; lambda1 ; beta1*Mw1 ; |A1-A2| ; |B1-B2|\n";

//    ofs << "No; beta0*(1-Mw0) ; lambda0 ; beta0*Mw0 ; beta1*(1-Mw1) ; lambda1 ; beta1*Mw1 ; |A1-A2| ; |B1-B2|\n";

    for(int64_t numFish = numFishMin; numFish <= numFishMax; numFish += fishStep) {

        Real total0 = numFish*numFishEggs;
        Real lambda0_I = groupI.Nout0 / (total0 * factorI),
             lambda0_II = groupII.Nout0 / (total0 * factorII);

        {
        std::ofstream ofs("C:\\work\\LambdaSim_Fish" + std::to_string(numFish) + "_I.csv");
        fprintf(stderr, "###### numFish = %lld #######\n", numFish);
         ofs << "No; beta0*(1-Mw0) ; lambda0 ; beta0*Mw0 ; beta1*(1-Mw1) ; lambda1 ; beta1*Mw1 ; |A1-A2| ; |B1-B2|\n";
        //ofs << "NumFish=" << numFish << "\n";
        //ofs << "Group I\n";
        runLambdaInternal(ofs, groupI, lambda0_I, true);
        }

        {
        //ofs << "####### ; ########; #######;########;########;###########################\n";
        std::ofstream ofs("C:\\work\\LambdaSim_Fish" + std::to_string(numFish) + "_II.csv");
         ofs << "No; beta0*(1-Mw0) ; lambda0 ; beta0*Mw0 ; beta1*(1-Mw1) ; lambda1 ; beta1*Mw1 ; |A1-A2| ; |B1-B2|\n";
        //ofs << "Group II\n";
        runLambdaInternal(ofs, groupII, lambda0_II, false);
        }

        //std::flush(ofs);
    }

    exit(1);
}

}; // Simulator

int main(int argc, char *argv[])
{
    try {
        Simulator simi;
        simi.runLambda();
    }
    catch(std::exception& ex) {
        std::cerr << "Exception: " << ex.what() << std::endl;
    }
    return 0;
}
