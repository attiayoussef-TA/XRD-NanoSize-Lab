#include "xrd_analysis.hpp"
#include <iostream>

int main(){
    std::cout << "================================================\n";
    std::cout << "      C++ XRD PHASE ANALYSIS TOOLKIT v1.0         \n";
    std::cout << "  Implementation: Levenberg-Marquardt (LDLT)    \n";
    std::cout << "            Initialize process                   \n";
    std::cout << "================================================\n";

    // Analyze with Cu-Kα source (1.5406 Å)
    analyze("2017","data/alumina_2017.csv",1.5406);
    analyze("2024","data/alumina_2024.csv",1.5406);

    std::cout<<"\nAnalysis complete.\n";
    return 0;
}
