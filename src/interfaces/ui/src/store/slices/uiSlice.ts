import { createSlice, PayloadAction } from '@reduxjs/toolkit';

interface UIState {
  activeView: 'dashboard' | 'codeEditor' | 'knowledgeExplorer' | 'taskManager' | 'settings';
  theme: 'jarvis' | 'light' | 'highContrast' | 'custom';
  sidebarOpen: boolean;
  contextPanelOpen: boolean;
  notifications: Array<{
    id: string;
    message: string;
    type: 'info' | 'success' | 'warning' | 'error';
    timestamp: number;
    read: boolean;
  }>;
  isFullscreen: boolean;
  modalOpen: {
    settings: boolean;
    help: boolean;
    newTask: boolean;
    newCode: boolean;
    [key: string]: boolean;
  };
}

const initialState: UIState = {
  activeView: 'dashboard',
  theme: 'jarvis',
  sidebarOpen: true,
  contextPanelOpen: false,
  notifications: [],
  isFullscreen: false,
  modalOpen: {
    settings: false,
    help: false,
    newTask: false,
    newCode: false,
  },
};

export const uiSlice = createSlice({
  name: 'ui',
  initialState,
  reducers: {
    setActiveView: (
      state,
      action: PayloadAction<UIState['activeView']>
    ) => {
      state.activeView = action.payload;
    },
    setTheme: (state, action: PayloadAction<UIState['theme']>) => {
      state.theme = action.payload;
    },
    toggleSidebar: (state) => {
      state.sidebarOpen = !state.sidebarOpen;
    },
    setSidebarOpen: (state, action: PayloadAction<boolean>) => {
      state.sidebarOpen = action.payload;
    },
    toggleContextPanel: (state) => {
      state.contextPanelOpen = !state.contextPanelOpen;
    },
    setContextPanelOpen: (state, action: PayloadAction<boolean>) => {
      state.contextPanelOpen = action.payload;
    },
    addNotification: (
      state,
      action: PayloadAction<Omit<UIState['notifications'][0], 'read' | 'timestamp'>>
    ) => {
      state.notifications.push({
        ...action.payload,
        read: false,
        timestamp: Date.now(),
      });
    },
    markNotificationAsRead: (state, action: PayloadAction<string>) => {
      const notification = state.notifications.find(n => n.id === action.payload);
      if (notification) {
        notification.read = true;
      }
    },
    clearNotifications: (state) => {
      state.notifications = [];
    },
    toggleFullscreen: (state) => {
      state.isFullscreen = !state.isFullscreen;
    },
    setFullscreen: (state, action: PayloadAction<boolean>) => {
      state.isFullscreen = action.payload;
    },
    openModal: (state, action: PayloadAction<string>) => {
      state.modalOpen[action.payload] = true;
    },
    closeModal: (state, action: PayloadAction<string>) => {
      state.modalOpen[action.payload] = false;
    },
    resetUI: () => initialState,
  },
});

export const {
  setActiveView,
  setTheme,
  toggleSidebar,
  setSidebarOpen,
  toggleContextPanel,
  setContextPanelOpen,
  addNotification,
  markNotificationAsRead,
  clearNotifications,
  toggleFullscreen,
  setFullscreen,
  openModal,
  closeModal,
  resetUI,
} = uiSlice.actions;

export default uiSlice.reducer;
