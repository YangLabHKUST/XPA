// Mixture Model Net
// Copyright (C) 2019  Shunkang Zhang
// Copyright (C) 2014-2017  Harvard University
//
// This program is free software: you can redistribute it and/or modify
// it under the terms of the GNU General Public License as published by
// the Free Software Foundation, either version 3 of the License, or
// (at your option) any later version.

// This program is distributed in the hope that it will be useful,
//    but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
// GNU General Public License for more details.

// You should have received a copy of the GNU General Public License
// along with this program.  If not, see <https://www.gnu.org/licenses/>.

#ifndef LMMNET_PARAMETERS_H
#define LMMNET_PARAMETERS_H

#include <vector>
#include <string>
#include <utility>

#include <boost/program_options.hpp>

#include "TypeDef.h"

namespace LMMNET {
class Params {

 public:

  // for genotype input
  std::string famFile, predfamFile, auxfamFile;
  std::vector<std::string> bimFiles, bedFiles; // allows to input multiple files
  std::vector<std::string> predbimFiles, predbedFiles;
  std::vector<std::string> auxbimFiles, auxbedFiles;

  std::vector<std::string> removeIndivsFiles; // indivs to remove
  std::vector<std::string> auxremoveIndivsFiles;
  std::vector<std::string> predremoveIndivsFiles;
  std::vector<std::string> removeSnpsFiles, auxremoveSnpsFiles; // SNPs to remove
  std::vector<std::string> predremoveSnpsFiles; // Todo: implement the remove function in prediction files

  // for phenotype input
  std::string phenoFile; // allow only one phenotype file
  std::string auxphenoFile; // store aux phenotype data
  std::vector<std::string> phenoCols; // selected phenotype to analysis (Todo: allow multiple phenotype analysis)
  std::vector<std::string> auxphenoCols; // selected auxiliary phenotype to analysis
  bool phenoUseFam; // use Fam 6th col as phenotype

  // for covariate input
  std::string covarFile; // allow only one covariate file
  std::string predcovarFile;
  std::string auxcovarFile;
  std::vector<std::string> covarCols, auxcovarCols,
      precovarCols; // selected covariates to analysis (allow multiple Cols)

  // for setting algorithm
  int numThreads;
  int numChrom;
  int numCalibSnps;
  int snpsPerBlock; // split snps into small blocks to analysis

  uint64 estIterationTrace; // number of iteration involved in estimating main dataset trace
  uint64 estIterationTraceAux; // number of iteration involved in estimating auxiliary dataset trace
  uint64 estIterationDelta; // number of iteration involved in estimating generic correlation
  uint64 maxIterationConj; // number of max iteration in conjugate gradient
  double convergenceLevel; // convergence level of conjugate gradient
  bool useExactTrace;
  bool associationTest;
  bool prediction;
  bool geneticCorr;
  bool samplesCalibrationFactor;
  bool useApproFixEffect;

  std::string imputeMethod;

  // for setting quality control
  double maxMissingPerSnp, maxMissingPerIndiv;
  int maxModelSnps; // the maximum number of snps analysis

  // for final output
  std::string outputFile;

  // for store temporary input templates
  std::vector<std::string> bimFileTemplates, bedFileTemplates;
  std::vector<std::string> predbimFileTemplates, predbedFileTemplates;
  std::vector<std::string> auxbimFileTemplates, auxbedFileTemplates;
  std::vector<std::string> removeIndivFileTemplates, removeSnpsFileTemplates;
  std::vector<std::string> auxremoveIndivFileTemplates, auxremoveSnpsFileTemplates;
  std::vector<std::string> covarColTemplates, auxcovarColTemplates, precovarColTemplates;

  bool processCommandLineArgs(int argc, char **argv);

 private:
  const std::string DELITMS = "{:}";

  // check invalid argument
  bool checkArguments(boost::program_options::command_line_parser &cmd,
                      boost::program_options::options_description &opt,
                      boost::program_options::variables_map &vm);

  // deal with input template files
  std::vector<std::string> analysisTemplates(const std::vector<std::string> &templatesFile);
  std::vector<std::string> analysisTemplate(const std::string &templateFile);
  std::string findDelimiters(const std::string &str, const std::string &delim);
  std::vector<std::string> takeMutipleDelimiters(const std::string &str, const std::string &delim);

  int stoi(const std::string &s);
  std::string itos(int i);

  void rangeErrorExit(const std::string &str, const std::string &delims);
};
}
#endif //LMMNET_PARAMETERS_H
