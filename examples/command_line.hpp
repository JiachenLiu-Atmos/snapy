#pragma once

namespace snap {

//! \brief Command line argument parser and container
//!
//! This class parses and stores command line arguments for the simulation.
//! It follows the Singleton pattern to ensure a single instance across the
//! application.
class CommandLine {
 protected:
  //! \brief Protected constructor
  //!
  //! \param[in] argc Number of command line arguments
  //! \param[in] argv Array of command line argument strings
  CommandLine(int argc, char **argv);

 public:
  char *input_filename = nullptr;    //!< Input configuration filename
  char *restart_filename = nullptr;  //!< Restart file name
  char *prundir = nullptr;           //!< Run directory path
  int res_flag;                      //!< Restart flag
  int narg_flag;                     //!< Number of arguments flag
  int iarg_flag;                     //!< Argument index flag
  int mesh_flag;                     //!< Mesh flag
  int wtlim;                         //!< Wall time limit
  int argc;                          //!< Number of command line arguments
  int nthreads;                      //!< Number of threads to use
  char **argv;                       //!< Command line argument strings

  //! \brief Parse command line arguments
  //!
  //! Static factory method to parse arguments and create CommandLine instance.
  //!
  //! \param[in] argc Number of command line arguments
  //! \param[in] argv Array of command line argument strings
  //! \return Pointer to CommandLine singleton instance
  static CommandLine *ParseArguments(int argc, char **argv);

  //! \brief Get the singleton instance
  //!
  //! \return Pointer to CommandLine singleton instance, or nullptr if not
  //! initialized
  static CommandLine *GetInstance();

  //! \brief Destroy the singleton instance
  static void Destroy();

  //! \brief Destructor
  ~CommandLine() {}

 private:
  static CommandLine *mycli_;  //!< Singleton instance pointer
};

}  // namespace snap
