import React, { useEffect, useState } from 'react';
import { ThemeProvider, CssBaseline } from '@mui/material';
import { Routes, Route, Navigate } from 'react-router-dom';
import { useSelector } from 'react-redux';
import getTheme from './styles/themes';
import { RootState } from './store';
import MainLayout from './layouts/MainLayout';
import Dashboard from './components/dashboard/Dashboard';
import CodeEditor from './components/codeEditor/CodeEditor';
import KnowledgeExplorer from './components/knowledgeExplorer/KnowledgeExplorer';
import TaskManager from './components/taskManager/TaskManager';
import Settings from './components/settings/Settings';
import StartupScreen from './components/common/StartupScreen';
import NotFound from './components/common/NotFound';

const App: React.FC = () => {
  const { theme } = useSelector((state: RootState) => state.ui);
  const { status } = useSelector((state: RootState) => state.system);
  const [loading, setLoading] = useState(true);

  // Simulate startup process
  useEffect(() => {
    const timer = setTimeout(() => {
      setLoading(false);
    }, 3000);

    return () => clearTimeout(timer);
  }, []);

  if (loading || status === 'initializing') {
    return <StartupScreen />;
  }

  return (
    <ThemeProvider theme={getTheme(theme)}>
      <CssBaseline />
      <div className={`theme-${theme}`}>
        <Routes>
          <Route path="/" element={<MainLayout />}>
            <Route index element={<Navigate to="/dashboard" replace />} />
            <Route path="dashboard" element={<Dashboard />} />
            <Route path="code-editor" element={<CodeEditor />} />
            <Route path="knowledge-explorer" element={<KnowledgeExplorer />} />
            <Route path="task-manager" element={<TaskManager />} />
            <Route path="settings" element={<Settings />} />
            <Route path="*" element={<NotFound />} />
          </Route>
        </Routes>
      </div>
    </ThemeProvider>
  );
};

export default App;
