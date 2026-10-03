/**
 * @file upower_stub.c
 * @brief upower-glib stub implementation (see upower.h).
 *
 * Never returns a client, so lpmd skips the upowerd signals.
 */
#include "upower.h"

/** @brief See upower.h: up_client_new_full(). */
UpClient *up_client_new_full(GCancellable *cancellable, GError **error) {
    (void)cancellable;
    g_set_error(error, G_IO_ERROR, G_IO_ERROR_NOT_SUPPORTED,
                "upower-glib stub (WinHint conda-only build): battery state not monitored, assuming AC power");
    return NULL;
}

/** @brief See upower.h: up_client_get_on_battery(). */
gboolean up_client_get_on_battery(UpClient *client) {
    (void)client;
    return FALSE;
}

/** @brief See upower.h: up_client_get_devices2(). */
GPtrArray *up_client_get_devices2(UpClient *client) {
    (void)client;
    return g_ptr_array_new();
}
