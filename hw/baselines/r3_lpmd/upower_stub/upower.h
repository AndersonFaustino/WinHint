/**
 * @file upower.h
 * @brief Minimal upower-glib stand-in for building intel-lpmd with conda tools only
 * (WinHint R3, docs/guide/hardware/index.md §3.R3). upower-glib has no conda package; intel-lpmd uses it
 * only to learn whether the machine runs on battery (src/lpmd_proc.c). This stub behaves
 * like "upowerd not reachable": up_client_new_full() fails with a GError, and
 * up_client_get_on_battery() reports AC power (FALSE). The campaign runs on AC power, which
 * is what the real upowerd would report. Deviation recorded in WINHINT_LPMD_DEVIATIONS.
 */
#ifndef WINHINT_UPOWER_STUB_H
#define WINHINT_UPOWER_STUB_H
#include <glib-object.h>
#include <gio/gio.h>

G_BEGIN_DECLS
/** @brief Opaque upower client (never instantiated by the stub). */
typedef struct _UpClient UpClient;
/** @brief Opaque upower device (declared for API compatibility). */
typedef struct _UpDevice UpDevice;

/**
 * @brief Stub constructor: always fails.
 * @param[in]  cancellable Ignored.
 * @param[out] error       Set to G_IO_ERROR_NOT_SUPPORTED (if non-NULL).
 * @return Always NULL.
 */
UpClient  *up_client_new_full(GCancellable *cancellable, GError **error);
/**
 * @brief Stub battery query: reports AC power.
 * @param[in] client Ignored.
 * @return Always FALSE.
 */
gboolean   up_client_get_on_battery(UpClient *client);
/**
 * @brief Stub device list.
 * @param[in] client Ignored.
 * @return A new empty GPtrArray (caller owns it).
 */
GPtrArray *up_client_get_devices2(UpClient *client);
G_END_DECLS
#endif
