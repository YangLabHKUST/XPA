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


#include <cstdlib>
#include <cmath>
#include <vector>
#include <string>
#include <sstream>
#include <set>
#include <algorithm>
#include <utility>
#include <numeric>
#include <iostream>

#include <boost/program_options.hpp>

#include "IOUtils.h"
#include "Parameters.h"

namespace LMMNET {

using std::vector;
using std::string;
using std::cout;
using std::cerr;
using std::endl;
using namespace boost::program_options;

bool Params::processCommandLineArgs(int argc, char **argv) {

  options_description opts("params");
  opts.add_options()
      ("help, h", "print help message to screen")

      // for genotype input
      ("bfile", value<string>(), "prefix of PLINK .fam, .bim, .bed files")
      ("fam", value<string>(&famFile), "PLINK .fam file")
      ("bim", value<vector<string> >(&bimFileTemplates), "PLINK .bim file(s) (allow templates input)")
      ("bed", value<vector<string> >(&bedFileTemplates), "PLINK .bed file(s) (allow templates input)")
      ("auxbfile", value<string>(), "prefix of PLINK for auxiliary .fam .bim. bed files")
      ("auxfam", value<string>(&auxfamFile), "PLINK auxiliary .fam file")
      ("auxbim", value<vector<string> >(&auxbimFileTemplates), "PLINK auxiliary .bim file(s) (allow templates input)")
      ("auxbed", value<vector<string> >(&auxbedFileTemplates), "PLINK auxiliary .bed file(s) (allow templates input)")
      ("predbfile", value<string>(), "prefix of PLINK for prediction .fam .bim. bed files")
      ("predfam", value<string>(&predfamFile), "PLINK prediction .fam file")
      ("predbim", value<vector<string> >(&predbimFileTemplates), "PLINK prediction .bim file(s) (allow templates input)")
      ("predbed", value<vector<string> >(&predbedFileTemplates), "PLINK prediction .bed file(s) (allow templates input)")
      ("removeIndiv", value<vector<string> >(&removeIndivFileTemplates), "select individuals removed from model")
      ("auxremoveIndiv",
       value<vector<string> >(&auxremoveIndivFileTemplates),
       "select individuals removed from auxiliary model")
      ("removeSnps", value<vector<string> >(&removeSnpsFileTemplates), "select SNPs removed from model")
      ("auxremoveSnps", value<vector<string> >(&auxremoveSnpsFileTemplates), "select SNPs removed from auxiliary model")
      ("modelSnps", value<vector<string> >(&modelSnpsFileTemplates), "select SNPs removed from model")
      ("auxmodelSnps", value<vector<string> >(&auxmodelSnpsFileTemplates), "select SNPs removed from auxiliary model")
      // for phenotype input
      ("phenoFile", value<string>(&phenoFile), "phenotype file (header required)")
      ("auxphenoFile", value<string>(&auxphenoFile), "aux phenotype file (header required)")
      ("phenoCol", value<vector<string> >(&phenoCols), "select phenotype column include in model")
      ("auxphenoCol", value<vector<string> >(&auxphenoCols), "select auxiliary phenotype column include in model")
      ("phenoUseFam", "whether use the 6th col as phenotype")
      //("associationTest", value<bool>(&associationTest)->default_value(false), "whether compute the association test")
      ("prediction", value<bool>(&prediction)->default_value(false), "whether predict the new data (one dataset only)")
      ("geneticCorr",
       value<bool>(&geneticCorr)->default_value(false),
       "whether predict the new data by using genetic correlation (two datasets required)")

      // for covariate input
      ("covarFile", value<string>(&covarFile), "covariate file (header required)")
      ("predcovarFile", value<string>(&predcovarFile), "covariate file for prediction (header required)")
      ("auxcovarFile", value<string>(&auxcovarFile), "covariate file for auxiliary dataset (header required)")
      ("covarCol",
       value<vector<string> >(&covarColTemplates),
       "select covariate columns included in model (default all cols)")
      ("auxcovarCol",
       value<vector<string> >(&auxcovarColTemplates),
       "select covariate columns included in auxiliary model (default all cols)")
      ("precovarCol",
       value<vector<string> >(&precovarColTemplates),
       "select covariate columns included in prediction model (default all cols)")

      // for algorithm
      ("numChrom", value<int>(&numChrom)->default_value(22), "the number of chromsome in model (for testing association)")
      ("numThreads", value<int>(&numThreads)->default_value(1), "number of computational threads")
      ("snpsPerBlock", value<int>(&snpsPerBlock)->default_value(64), "the number of snps per block in computation")
      ("estIteration", value<uint64>(&estIterationTrace)->default_value(10),
       "the number of sample iteration in estimating trace by mom estimator")
      ("estIterationAux", value<uint64>(&estIterationTraceAux)->default_value(10),
       "the number of sample iteration in estimating trace by mom estimator")
      ("estIterationDelta", value<uint64>(&estIterationDelta)->default_value(10),
       "the number of sample iteration in estimating trace by mom estimator")
      ("maxIterationConj", value<uint64>(&maxIterationConj)->default_value(100),
       "the number of maxiteration in computation of conjugate gradient")
      ("convergenceLevel",
       value<double>(&convergenceLevel)->default_value(1e-5),
       "convergence level of conjugate gradient")
      ("maxMissingPerSnp", value<double>(&maxMissingPerSnp)->default_value(0.1, "0.1"), "QC: max missing rate per SNP")
      ("maxMissingPerIndiv",
       value<double>(&maxMissingPerIndiv)->default_value(0.1, "0.1"),
       "QC: max missing rate per person")
      ("maxModelSnps", value<int>(&maxModelSnps)->default_value(1000000), "the maximum number of snps in model")
      ("useExactTrace", value<bool>(&useExactTrace)->default_value(false), "whether compute the exact trace")
      ("useApproFixEffect", value<bool>(&useApproFixEffect)->default_value(false), "whether use the approximate fix effect to reduce computation time")
      ("imputeMethod", value<string>(&imputeMethod)->default_value("mean"), "The way to impute missing value. Either using mean value or zero")
      ("RAMeff", value<bool>(&RAMeff)->default_value(false), "Whether use RAM efficient model")

      ("outputFile", value<string>(&outputFile)->default_value("./result.txt"), "output file for the final result");

  variables_map vm;
  command_line_parser cmd(argc, argv);
  cmd.options(opts);
  // allow to guess command from abbreviated spelling
  cmd.style(command_line_style::default_style ^ command_line_style::allow_guessing);

  bool ifValid = checkArguments(cmd, opts, vm);
  return ifValid;
}

bool Params::checkArguments(boost::program_options::command_line_parser &cmd,
                            boost::program_options::options_description &opt,
                            boost::program_options::variables_map &vm) {
  try {
    store(cmd.run(), vm);

    if (vm.count("help")) {
      cout << opt << endl;
      exit(0);
    }

    notify(vm); // update arguments

    if (vm.count("bfile") + (vm.count("fam") || vm.count("bim") || vm.count("bed")) != 1) {
      cerr << "Error: Cannot specifiy --bfile and --fam or --bim or --bed together" << endl;
      return false;
    }

    if (vm.count("bfile")) {
      string bfile = vm["bfile"].as<string>();
//      famFile = bfile + ".fam";
      bimFileTemplates.push_back(bfile + ".bim");
      bedFileTemplates.push_back(bfile + ".bed");
    }

    if (vm.count("auxbfile")) {
      string bfile = vm["auxbfile"].as<string>();
      auxfamFile = bfile + ".fam";
      auxbimFileTemplates.push_back(bfile + ".bim");
      auxbedFileTemplates.push_back(bfile + ".bed");
    }

    if (vm.count("predbfile")) {
      string predbfile = vm["predbfile"].as<string>();
      predfamFile = predbfile + ".fam";
      predbimFileTemplates.push_back(predbfile + ".bim");
      predbedFileTemplates.push_back(predbfile + ".bed");
    }

    phenoUseFam = vm.count("phenoUseFam");

    if (bimFileTemplates.empty()) {
      cerr << "Error: Please specifiy bim file"
           << endl;
      return false;
    }
    if (bedFileTemplates.empty()) {
      cerr << "Error: Please specifiy bed file"
           << endl;
      return false;
    }
    if (bimFileTemplates.size() != bedFileTemplates.size()) {
      cerr << "Error: Numbers of bim files and bed files must match" << endl;
      return false;
    }

    bimFiles = analysisTemplates(bimFileTemplates);
    bedFiles = analysisTemplates(bedFileTemplates);
    // copy and change the name
    if (vm.count("bfile")) {
      famFile = bimFiles[0];
      famFile.replace(famFile.end() - 3, famFile.end(), "fam");
    }

    if (famFile.empty()) {
      cerr << "Error: Please specifiy fam file"
           << endl;
      return false;
    }

    if (bimFiles.size() != bedFiles.size()) {
      cerr << "Error: Numbers of bim files and bed files must match" << endl;
      return false;
    }

    // deal with the template input
    predbedFiles = analysisTemplates(predbedFileTemplates);
    predbimFiles = analysisTemplates(predbimFileTemplates);
    auxbedFiles = analysisTemplates(auxbedFileTemplates);
    auxbimFiles = analysisTemplates(auxbimFileTemplates);
    removeIndivsFiles = analysisTemplates(removeIndivFileTemplates);
    removeSnpsFiles = analysisTemplates(removeSnpsFileTemplates);
    auxremoveIndivsFiles = analysisTemplates(auxremoveIndivFileTemplates);
    auxremoveSnpsFiles = analysisTemplates(auxremoveSnpsFileTemplates);
    modelSnpsFiles = analysisTemplates(modelSnpsFileTemplates);
    auxmodelSnpsFiles = analysisTemplates(auxmodelSnpsFileTemplates);
    covarCols = analysisTemplates(covarColTemplates);
    auxcovarCols = analysisTemplates(auxcovarColTemplates);
    precovarCols = analysisTemplates(precovarColTemplates);

    if (predbedFiles.empty() && prediction) {
      cerr << "Error: Missing prediction bedFile and please specify it" << endl;
      return false;
    }

    if (phenoFile.empty() != phenoCols.empty()) {
      cerr << "Error: Please specify phenoFile and phenoCols at the same time" << endl;
      return false;
    }

    if (!phenoFile.empty() && phenoUseFam) {
      cerr << "Error: If phenoFile is provided, phenoUseFam cannot be set" << endl;
      return false;
    }

    if (covarFile.empty()) {
      cout << "Warning: Missing covariate files" << endl;
      if (!predcovarFile.empty()) {
        cerr << "Error: The training model does not have fixed effect \n";
        cerr << "You cannot make prediction with fixed effect" << endl;
        return false;
      }
    }

    if (predcovarFile.empty()) {
      cout << "Warning: Missing prediciton covariate file " << endl;
    }

    if (covarFile.empty() && !covarCols.empty()) {
      cerr << "Error: If covarCols are provided, covarFile must be provided" << endl;
      return false;
    }

    if (convergenceLevel > 1e-5) {
      cerr << "Error: convergence level involved in conjugate gradient should be smaller than 1e-5" << endl;
      return false;
    }

    if (!(0 <= maxMissingPerSnp && maxMissingPerSnp <= 1)) {
      cerr << "Error: maxMissingPerSnp must be between 0 and 1" << endl;
      return false;
    }
    if (!(0 <= maxMissingPerIndiv && maxMissingPerIndiv <= 1)) {
      cerr << "Error: maxMissingPerIndiv must be between 0 and 1" << endl;
      return false;
    }

    if (prediction || geneticCorr) {
      FileUtils::requireEmptyOrWriteable(outputFile+"_posteriorMean.txt");
      FileUtils::requireEmptyOrWriteable(outputFile+"_fixeff.txt");
      FileUtils::requireEmptyOrWriteable(outputFile+"__predict.txt");
    }

    if (associationTest) {
      FileUtils::requireEmptyOrWriteable(outputFile);
    }

    // check input and output file status
    FileUtils::requireEmptyOrReadable(famFile);
    FileUtils::requireEachEmptyOrReadable(bimFiles);
    FileUtils::requireEachEmptyOrReadable(bedFiles);
    FileUtils::requireEachEmptyOrReadable(predbedFiles);
    FileUtils::requireEachEmptyOrReadable(auxbedFiles);
    FileUtils::requireEachEmptyOrReadable(auxbimFiles);
    FileUtils::requireEachEmptyOrReadable(removeIndivsFiles);
    FileUtils::requireEachEmptyOrReadable(removeSnpsFiles);
    FileUtils::requireEachEmptyOrReadable(auxremoveIndivsFiles);
    FileUtils::requireEachEmptyOrReadable(auxremoveSnpsFiles);
    FileUtils::requireEachEmptyOrReadable(predremoveIndivsFiles);
    FileUtils::requireEachEmptyOrReadable(predremoveSnpsFiles);
    FileUtils::requireEachEmptyOrReadable(modelSnpsFiles);
    FileUtils::requireEachEmptyOrReadable(auxmodelSnpsFiles);
    FileUtils::requireEmptyOrReadable(phenoFile);
    FileUtils::requireEmptyOrReadable(auxphenoFile);
    FileUtils::requireEmptyOrReadable(covarFile);
    FileUtils::requireEmptyOrReadable(auxcovarFile);
    FileUtils::requireEmptyOrReadable(predcovarFile);
//    FileUtils::requireEmptyOrWriteable(outputFile);

  }
  catch (error &e) {
    cerr << "Error: " << e.what() << endl << endl;
    cerr << opt << endl;
    return false;
  }
  return true;
}

std::vector<std::string> Params::analysisTemplates(const std::vector<std::string> &templatesFile) {
  vector<string> extractedTamplates;
  for (uint i = 0; i < templatesFile.size(); i++) {
    vector<string> range = analysisTemplate(templatesFile[i]);
    extractedTamplates.insert(extractedTamplates.end(), range.begin(), range.end());
  }
  return extractedTamplates;
}

std::vector<std::string> Params::analysisTemplate(const std::string &templateFile) {
  vector<string> ret;
  string delims = findDelimiters(templateFile, DELITMS);
  if (delims.empty())
    ret.push_back(templateFile);
  else if (delims == DELITMS) {
    vector<string> tokens = takeMutipleDelimiters(templateFile, DELITMS);
    for (int i = 0; i < (int) templateFile.size(); i++)
      if (templateFile[i] == ':' && (templateFile[i - 1] == '{' || templateFile[i + 1] == '}'))
        rangeErrorExit(templateFile, delims);
    int startInd = (templateFile[0] != DELITMS[0]), endInd = startInd + 1;
    string prefix, suffix;
    if (templateFile[0] != DELITMS[0]) prefix = tokens[0];
    if (templateFile[templateFile.length() - 1] != DELITMS[2]) suffix = tokens.back();
    int start = stoi(tokens[startInd]), end = stoi(tokens[endInd]);
    if (start > end + 1 || end > start + 1000000) {
      cerr << "ERROR: Invalid range in template string: " << templateFile << endl;
      cerr << "  Start: " << start << endl;
      cerr << "  End: " << end << endl;
      exit(1);
    }
    for (int i = start; i <= end; i++)
      ret.push_back(prefix + itos(i) + suffix);
  } else
    rangeErrorExit(templateFile, delims);
  return ret;

}

std::string Params::findDelimiters(const std::string &str, const std::string &delim) {
  string delims;
  for (uint c = 0; c < str.length(); c++) {
    if (delim.find(str[c], 0) != string::npos)
      delims += str[c];
  }
  return delims;
}

std::vector<std::string> Params::takeMutipleDelimiters(const std::string &str, const std::string &delim) {
  uint p = 0;
  vector<string> ans;
  string temp;
  while (p < str.length()) {
    temp = "";
    while (p < str.length() && delim.find(str[p], 0) != string::npos)
      p++;
    while (p < str.length() && delim.find(str[p], 0) == string::npos) {
      temp += str[p];
      p++;
    }
    if (temp != "")
      ans.push_back(temp);
  }
  return ans;
}

std::string Params::itos(int i) {
  std::ostringstream oss;
  oss << i;
  return oss.str();
}

int Params::stoi(const string &s) {
  int i;
  if (sscanf(s.c_str(), "%d", &i) == 0) {
    cerr << "ERROR: Could not parse integer from string: " << s << endl;
    exit(1);
  }
  return i;
}

void Params::rangeErrorExit(const string &str, const string &delims) {
  cerr << "ERROR: Invalid delimiter sequence for specifying range: " << endl;
  cerr << "  Template string: " << str << endl;
  cerr << "  Delimiter sequence found: " << delims << endl;
  cerr << "Range in must have format {start:end} with no other " << DELITMS
       << " chars" << endl;
  exit(1);
}
}

