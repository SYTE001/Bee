import { useCallback, useEffect, useState } from "react";

import { api, ApiError, getToken, setToken } from "../api/client";
import type { User } from "../api/types";

export function useAuth() {
  const [user, setUser] = useState<User | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  const refresh = useCallback(async () => {
    if (!getToken()) {
      setUser(null);
      setLoading(false);
      return;
    }
    try {
      const me = await api.me();
      setUser(me);
      setError(null);
    } catch (err) {
      if (err instanceof ApiError && err.status === 401) {
        setToken(null);
        setUser(null);
      } else {
        setError(err instanceof Error ? err.message : "auth failed");
      }
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => {
    void refresh();
  }, [refresh]);

  const login = useCallback(
    async (email: string, password: string) => {
      const res = await api.login(email, password);
      setToken(res.access_token);
      setUser(res.user);
      return res.user;
    },
    [],
  );

  const register = useCallback(
    async (email: string, password: string, displayName?: string) => {
      const res = await api.register(email, password, displayName);
      setToken(res.access_token);
      setUser(res.user);
      return res.user;
    },
    [],
  );

  const logout = useCallback(() => {
    setToken(null);
    setUser(null);
  }, []);

  return { user, loading, error, login, register, logout, refresh };
}
