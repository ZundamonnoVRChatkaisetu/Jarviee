import { createSlice, PayloadAction } from '@reduxjs/toolkit';

interface SystemState {
  status: 'initializing' | 'ready' | 'error' | 'maintenance';
  connected: boolean;
  activeModules: string[];
  resourceUsage: {
    cpu: number;
    memory: number;
    storage: number;
  };
  lastActivity: number;
  apiVersion: string;
  activeProcesses: Array<{
    id: string;
    name: string;
    status: 'running' | 'paused' | 'completed' | 'error';
    progress: number;
    startTime: number;
    endTime?: number;
  }>;
}

const initialState: SystemState = {
  status: 'initializing',
  connected: false,
  activeModules: [],
  resourceUsage: {
    cpu: 0,
    memory: 0,
    storage: 0,
  },
  lastActivity: Date.now(),
  apiVersion: '0.1.0',
  activeProcesses: [],
};

export const systemSlice = createSlice({
  name: 'system',
  initialState,
  reducers: {
    setStatus: (state, action: PayloadAction<SystemState['status']>) => {
      state.status = action.payload;
    },
    setConnected: (state, action: PayloadAction<boolean>) => {
      state.connected = action.payload;
    },
    setActiveModules: (state, action: PayloadAction<string[]>) => {
      state.activeModules = action.payload;
    },
    addActiveModule: (state, action: PayloadAction<string>) => {
      if (!state.activeModules.includes(action.payload)) {
        state.activeModules.push(action.payload);
      }
    },
    removeActiveModule: (state, action: PayloadAction<string>) => {
      state.activeModules = state.activeModules.filter(
        (m) => m !== action.payload
      );
    },
    updateResourceUsage: (
      state,
      action: PayloadAction<Partial<SystemState['resourceUsage']>>
    ) => {
      state.resourceUsage = { ...state.resourceUsage, ...action.payload };
    },
    updateLastActivity: (state) => {
      state.lastActivity = Date.now();
    },
    setApiVersion: (state, action: PayloadAction<string>) => {
      state.apiVersion = action.payload;
    },
    addProcess: (
      state,
      action: PayloadAction<Omit<SystemState['activeProcesses'][0], 'startTime'>>
    ) => {
      state.activeProcesses.push({
        ...action.payload,
        startTime: Date.now(),
      });
    },
    updateProcess: (
      state,
      action: PayloadAction<{
        id: string;
        updates: Partial<SystemState['activeProcesses'][0]>;
      }>
    ) => {
      const { id, updates } = action.payload;
      const processIndex = state.activeProcesses.findIndex((p) => p.id === id);
      if (processIndex !== -1) {
        state.activeProcesses[processIndex] = {
          ...state.activeProcesses[processIndex],
          ...updates,
        };
      }
    },
    removeProcess: (state, action: PayloadAction<string>) => {
      state.activeProcesses = state.activeProcesses.filter(
        (p) => p.id !== action.payload
      );
    },
    resetSystem: () => initialState,
  },
});

export const {
  setStatus,
  setConnected,
  setActiveModules,
  addActiveModule,
  removeActiveModule,
  updateResourceUsage,
  updateLastActivity,
  setApiVersion,
  addProcess,
  updateProcess,
  removeProcess,
  resetSystem,
} = systemSlice.actions;

export default systemSlice.reducer;
