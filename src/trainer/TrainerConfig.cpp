#include "trainer/TrainerConfig.h"
#include <cmath>

// config presets
namespace {
    const TrainerConfig::Preset FAST_PRESET = {30, 20, 0.05, 0.8, 0.15, 0.2, 3}; // fast check
    const TrainerConfig::Preset PRECISE_PRESET = {200, 100, 0.005, 0.6, 0.05, 0.05, 7}; // more precise, more generations
}

// Constructor
TrainerConfig::TrainerConfig(const Preset& preset)
    : populationSize(preset.populationSize),
      generations(preset.generations),
      mutationRate(preset.mutationRate),
      crossoverRate(preset.crossoverRate),
      elitismRate(preset.elitismRate),
      mutationStrength(preset.mutationStrength),
      tournamentSize(preset.tournamentSize) {}

TrainerConfig TrainerConfig::fastConfig() {
    return TrainerConfig(FAST_PRESET);
}

TrainerConfig TrainerConfig::preciseConfig() {
    return TrainerConfig(PRECISE_PRESET);
}

// data validation
bool TrainerConfig::validate() const {
    if (populationSize == 0 || generations == 0 || mutationRate < 0.0 || mutationRate > 1.0 ||
         crossoverRate < 0.0 || crossoverRate > 1.0 || elitismRate < 0.0 ||
          elitismRate > 1.0 || mutationStrength < 0.0){
        return false;
    }
    return true;
}