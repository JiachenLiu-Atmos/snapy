// C/C++ headers
#include <csignal>

namespace snap {

class MeshBlockImpl;

//! \brief Signal handler for managing system signals
//!
//! Singleton class that handles system signals (SIGTERM, SIGINT, SIGALRM)
//! for graceful termination and wall-time limits in the simulation.
//! Used to catch interrupts and time-limit signals during long-running
//! simulations.
class SignalHandler {
 protected:
  //! \brief Protected constructor (Singleton pattern)
  SignalHandler();  // disable direct instantiation

 public:
  static constexpr int nsignal = 3;  //!< Number of signals handled
  static constexpr int ITERM = 0,    //!< SIGTERM signal index
      IINT = 1,                      //!< SIGINT signal index
      IALRM = 2;                     //!< SIGALRM signal index

  //! \brief Get the singleton instance
  //!
  //! \return Pointer to SignalHandler singleton instance
  static SignalHandler *GetInstance();

  //! \brief Destroy the singleton instance
  static void Destroy();

  //! \brief Set a signal flag (called by signal handlers)
  //!
  //! \param[in] s Signal index to set
  static void SetSignalFlag(int s);

  //! \brief Destructor
  ~SignalHandler() {}

  //! \brief Check all signal flags and take appropriate action
  //!
  //! \param[in] pmb Pointer to MeshBlock for context
  //! \return Signal status code
  int CheckSignalFlags(MeshBlockImpl const *pmb);

  //! \brief Get the value of a specific signal flag
  //!
  //! \param[in] s Signal index to query
  //! \return Signal flag value
  int GetSignalFlag(int s);

  //! \brief Set wall-time alarm
  //!
  //! \param[in] t Time in seconds for alarm
  void SetWallTimeAlarm(int t);

  //! \brief Cancel the wall-time alarm
  void CancelWallTimeAlarm();

 private:
  static SignalHandler *mysignal_;  //!< Singleton instance pointer
  static int signalflag_[nsignal];  //!< Array of signal flags
  static sigset_t mask_;            //!< Signal mask for blocking
};

}  // namespace snap
