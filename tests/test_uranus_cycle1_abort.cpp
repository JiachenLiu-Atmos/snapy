// Two defects, tested on different decks.
//
// column_finishes_the_two_cycles is RED on main 3f7ad96. examples/uranus.yaml
// aborts on cycle 1: five redos, cause "limiter", thetamin 0, then
// "Terminating abnormally". Smallest deck with that signature, including no
// extrapolate_ad warning: the shipped file, one x2 cell, 96 x1 cells, nlim 2.
// 92 x1 cells still prints one extrapolate_ad non-convergence, which the
// shipped 100 x 200 deck does not. Nine and fewer x1 cells also trip floor,
// clamp or saturation. Raising equation-of-state max-iter from 5 to 50
// clears the equilibrate_tp warnings on the full deck; cycle 1 still aborts.
// The abort does not follow from those warnings.
//
// abnormal_termination_exits_nonzero does not use that deck. A fix for the
// limiter abort makes the column finish, so the exit-code check needs a deck
// that aborts on purpose. test_abnormal_exit_floor.yaml sets pressure-floor
// above the initial pressure: every step trips the floor check and exhausts
// the redo budget. max_redo is not a yaml key on the pyharp this tree links
// (IntegratorOptionsImpl::from_yaml never reads it; the default stays 5), so
// a yaml max_redo of 0 would not force the abort. nlim is only a backstop.

// external
#include <gtest/gtest.h>

// C/C++
#include <sys/wait.h>
#include <unistd.h>

#include <string>
#include <vector>

namespace {

struct RunResult {
  int code;  // wait status translated to an exit code; 128+signal if signaled
  std::string log;
};

std::string find_file(std::vector<std::string> const& names) {
  for (auto const& name : names) {
    if (access(name.c_str(), F_OK) == 0) return name;
  }
  return {};
}

RunResult run_deck(std::string const& yaml_name) {
  auto bin = find_file(
      {std::getenv("SNAPY_RUN_HYDRO") ? std::getenv("SNAPY_RUN_HYDRO") : "",
       "../bin/run_hydro.release", "./run_hydro.release"});
  auto deck = find_file({yaml_name, "../tests/" + yaml_name});
  if (bin.empty() || deck.empty()) {
    return {-1, "missing run_hydro.release or the deck\n"};
  }

  int pipefd[2];
  EXPECT_EQ(pipe(pipefd), 0);
  pid_t pid = fork();
  EXPECT_GE(pid, 0);
  if (pid == 0) {
    dup2(pipefd[1], STDOUT_FILENO);
    dup2(pipefd[1], STDERR_FILENO);
    close(pipefd[0]);
    close(pipefd[1]);
    setenv("DEVICE", "cpu", 1);
    setenv("CUDA_VISIBLE_DEVICES", "", 1);
    execl(bin.c_str(), bin.c_str(), "-i", deck.c_str(),
          static_cast<char*>(nullptr));
    _exit(127);
  }
  close(pipefd[1]);
  std::string log;
  char buf[4096];
  ssize_t n = 0;
  while ((n = read(pipefd[0], buf, sizeof(buf))) > 0) log.append(buf, n);
  close(pipefd[0]);
  int status = 0;
  waitpid(pid, &status, 0);
  int code = WIFEXITED(status) ? WEXITSTATUS(status) : 128 + WTERMSIG(status);
  return {code, log};
}

}  // namespace

// The shipped failure, on the smallest column that keeps it. A fixed run with
// nlim 2 stops on the cycle limit instead of exhausting the redo budget.
TEST(UranusCycle1, column_finishes_the_two_cycles) {
  auto r = run_deck("test_uranus_cycle1_abort.yaml");
  ASSERT_NE(r.code, -1) << r.log;
  EXPECT_EQ(r.log.find("Maximum number of redo attempts exceeded"),
            std::string::npos);
  EXPECT_NE(r.log.find("Terminating on cycle limit"), std::string::npos);
}

// run_hydro breaks out of the time loop on a redo failure and then falls off
// main, so "Terminating abnormally" is reported as success. The floor deck
// aborts whatever the limiter does, so this stays red on a library that still
// exits 0 and goes green once finalize returns that status.
TEST(UranusCycle1, abnormal_termination_exits_nonzero) {
  auto r = run_deck("test_abnormal_exit_floor.yaml");
  SCOPED_TRACE(r.log);
  ASSERT_NE(r.code, -1) << r.log;
  ASSERT_NE(r.log.find("Terminating abnormally"), std::string::npos)
      << "this deck no longer terminates abnormally; the exit-code check was "
         "not exercised";
  EXPECT_NE(r.code, 0) << "Terminating abnormally exited 0";
}

// RED on 93d1de1. The round-off repairs no longer redo the 100 x 200 deck,
// and it is clean through cycle 38. A real H2S / H2S(l) transfer then does:
// 1.6e-10 kg/m^3, 1.8e-10 of the cell's gas density, about 200 times the
// 4096-ulp bound, across one whole horizontal row. The redo budget runs out
// at cycle 232. One x2 cell keeps that transfer (about 5e-9 of the cell) and
// exhausts at cycle 32. 80 x1 cells finish 50 cycles with no redo, and 50 x1
// still reaches nlim 40, so the vertical count stays 100. nx2 of 4 or 16
// dies in the first few cycles with limcut of order 1, which is a different
// failure. nlim 40 is past this column's abort.
TEST(UranusLate, column_reaches_cycle_40) {
  auto r = run_deck("test_uranus_late_abort.yaml");
  ASSERT_NE(r.code, -1) << r.log;
  EXPECT_EQ(r.code, 0) << r.log;
  // Kinetics must use the final, saturation-adjusted state. Reading the
  // previous stage's cloud abundance removes cloud that has already evaporated
  // and causes real species repairs, even before the redo budget is exhausted.
  EXPECT_EQ(r.log.find("Redoing the step"), std::string::npos) << r.log;
  EXPECT_EQ(r.log.find("Maximum number of redo attempts exceeded"),
            std::string::npos);
  EXPECT_EQ(r.log.find("Terminating abnormally"), std::string::npos);
  EXPECT_NE(r.log.find("Terminating on cycle limit"), std::string::npos);
}
