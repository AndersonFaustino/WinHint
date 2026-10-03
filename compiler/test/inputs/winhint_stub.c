/**
 * @file
 * @brief Stand-in for hw/libwinhint in call-mode tests: hints are no-ops.
 */
/** @brief No-op setwin runtime entry point. @param w Window size (ignored). */
void __winhint_setwin(unsigned w) { (void)w; }
/** @brief No-op region runtime entry point. @param id Region id (ignored). */
void __winhint_region(unsigned id) { (void)id; }
