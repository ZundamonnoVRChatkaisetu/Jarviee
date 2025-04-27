import { createSlice, PayloadAction } from '@reduxjs/toolkit';

export interface Message {
  id: string;
  content: string;
  role: 'user' | 'assistant' | 'system';
  timestamp: number;
  status?: 'sending' | 'sent' | 'error';
}

interface ChatState {
  messages: Message[];
  isTyping: boolean;
  context: Record<string, any>;
  conversationId: string | null;
  activeThread: string | null;
  threads: Array<{ id: string; title: string; lastUpdated: number }>;
}

const initialState: ChatState = {
  messages: [],
  isTyping: false,
  context: {},
  conversationId: null,
  activeThread: null,
  threads: [],
};

export const chatSlice = createSlice({
  name: 'chat',
  initialState,
  reducers: {
    addMessage: (state, action: PayloadAction<Message>) => {
      state.messages.push(action.payload);
    },
    updateMessage: (
      state,
      action: PayloadAction<{ id: string; updates: Partial<Message> }>
    ) => {
      const { id, updates } = action.payload;
      const messageIndex = state.messages.findIndex((msg) => msg.id === id);
      if (messageIndex !== -1) {
        state.messages[messageIndex] = {
          ...state.messages[messageIndex],
          ...updates,
        };
      }
    },
    setTyping: (state, action: PayloadAction<boolean>) => {
      state.isTyping = action.payload;
    },
    updateContext: (state, action: PayloadAction<Record<string, any>>) => {
      state.context = { ...state.context, ...action.payload };
    },
    setConversationId: (state, action: PayloadAction<string | null>) => {
      state.conversationId = action.payload;
    },
    setActiveThread: (state, action: PayloadAction<string | null>) => {
      state.activeThread = action.payload;
    },
    addThread: (
      state,
      action: PayloadAction<{ id: string; title: string; lastUpdated: number }>
    ) => {
      state.threads.push(action.payload);
    },
    updateThread: (
      state,
      action: PayloadAction<{
        id: string;
        updates: Partial<{ title: string; lastUpdated: number }>;
      }>
    ) => {
      const { id, updates } = action.payload;
      const threadIndex = state.threads.findIndex((thread) => thread.id === id);
      if (threadIndex !== -1) {
        state.threads[threadIndex] = {
          ...state.threads[threadIndex],
          ...updates,
        };
      }
    },
    clearMessages: (state) => {
      state.messages = [];
    },
    resetChat: () => initialState,
  },
});

export const {
  addMessage,
  updateMessage,
  setTyping,
  updateContext,
  setConversationId,
  setActiveThread,
  addThread,
  updateThread,
  clearMessages,
  resetChat,
} = chatSlice.actions;

export default chatSlice.reducer;
