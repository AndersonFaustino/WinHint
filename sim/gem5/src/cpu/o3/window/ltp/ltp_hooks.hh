/**
 * @file
 * @brief Narrow LTP hooks for commit.cc (ltpSetROB(), ltpCommit()).
 *
 * WinHint B9: Long-Term Parking -- narrow hooks used by commit.cc, so that
 * commit does not need the full LongTermParking definition.
 */

#ifndef __CPU_O3_WINDOW_LTP_LTP_HOOKS_HH__
#define __CPU_O3_WINDOW_LTP_LTP_HOOKS_HH__

#include "cpu/o3/dyn_inst_ptr.hh"

namespace gem5
{

namespace o3
{

class LongTermParking;
class ROB;

/**
 * @brief Commit::startupStage(): give the LTP the ROB (head = wake-up
 *        point).
 * @param ltp The LTP.
 * @param rob The ROB.
 */
void ltpSetROB(LongTermParking *ltp, ROB *rob);

/**
 * @brief Commit::commitHead(): train the urgency table (IBDA seeds).
 * @param ltp The LTP.
 * @param inst The committed instruction.
 */
void ltpCommit(LongTermParking *ltp, const DynInstPtr &inst);

} // namespace o3
} // namespace gem5

#endif // __CPU_O3_WINDOW_LTP_LTP_HOOKS_HH__
