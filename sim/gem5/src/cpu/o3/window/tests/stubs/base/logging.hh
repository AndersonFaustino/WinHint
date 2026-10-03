/**
 * @file
 * @brief Host-test stub of gem5's base/logging.hh.
 *
 * fatal()/panic() throw winhint_test::Fatal so that the tests can check
 * error paths; warn() is a no-op. Only the printf-like formatting the
 * window code uses is supported: every %-conversion prints the next
 * argument with operator<<.
 */
// Host-test stub of gem5's base/logging.hh: fatal()/panic() throw
// winhint_test::Fatal so that the tests can check error paths.
#ifndef WINHINT_TEST_STUB_BASE_LOGGING_HH
#define WINHINT_TEST_STUB_BASE_LOGGING_HH
#include <cctype>
#include <sstream>
#include <stdexcept>
#include <string>

namespace winhint_test
{
/** Exception thrown by the stubbed fatal() and panic(). */
struct Fatal : std::runtime_error
{
    using std::runtime_error::runtime_error;
};

/**
 * @brief Format base case: copy the rest of the format verbatim.
 * @param os Output.
 * @param f Remaining format string.
 */
inline void
fmtInto(std::ostringstream &os, const char *f)
{
    os << f;
}

/**
 * @brief Copy f to os, replacing the first conversion (% up to the next
 *        letter) with v and recursing for the rest; "%%" prints '%'.
 * @param os Output.
 * @param f Format string.
 * @param v Value for the first conversion.
 * @param r Values for the following conversions.
 */
template <typename T, typename... Rest>
void
fmtInto(std::ostringstream &os, const char *f, const T &v, const Rest &...r)
{
    for (; *f; ++f) {
        if (*f == '%' && f[1] == '%') { os << '%'; ++f; continue; }
        if (*f == '%') {
            ++f;
            while (*f && !std::isalpha((unsigned char)*f)) ++f;
            os << v;
            if (*f) ++f;
            fmtInto(os, f, r...);
            return;
        }
        os << *f;
    }
}

/**
 * @brief Throw Fatal("<kind>: <formatted message>").
 * @param kind "fatal" or "panic".
 * @param f Format string.
 * @param a Format arguments.
 */
template <typename... Args>
[[noreturn]] void
fail(const char *kind, const char *f, const Args &...a)
{
    std::ostringstream os;
    os << kind << ": ";
    fmtInto(os, f, a...);
    throw Fatal(os.str());
}
} // namespace winhint_test

/** Stub of gem5 fatal(): throws winhint_test::Fatal. */
#define fatal(...) ::winhint_test::fail("fatal", __VA_ARGS__)
/** Stub of gem5 panic(): throws winhint_test::Fatal. */
#define panic(...) ::winhint_test::fail("panic", __VA_ARGS__)
/** fatal() if c. */
#define fatal_if(c, ...) do { if (c) fatal(__VA_ARGS__); } while (0)
/** panic() if c. */
#define panic_if(c, ...) do { if (c) panic(__VA_ARGS__); } while (0)
/** Stub of gem5 warn(): no-op. */
#define warn(...) do {} while (0)
#endif
