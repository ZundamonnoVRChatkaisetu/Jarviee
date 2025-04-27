import React from 'react';
import { 
  Snackbar, 
  Alert, 
  AlertTitle,
  Box
} from '@mui/material';
import { useSelector, useDispatch } from 'react-redux';
import { RootState } from '../../store';
import { markNotificationAsRead } from '../../store/slices/uiSlice';
import { styled } from '@mui/material/styles';
import { motion, AnimatePresence } from 'framer-motion';

const NotificationContainer = styled(Box)(({ theme }) => ({
  position: 'fixed',
  bottom: 16,
  right: 16,
  zIndex: theme.zIndex.snackbar,
  display: 'flex',
  flexDirection: 'column',
  alignItems: 'flex-end',
  maxWidth: '100%',
  maxHeight: '80vh',
  overflow: 'hidden',
  pointerEvents: 'none', // Allow clicks to pass through container
}));

const NotificationItem = styled(motion.div)({
  marginBottom: 8,
  width: '100%',
  maxWidth: 400,
  pointerEvents: 'auto', // Restore pointer events for each notification
});

interface ToastNotificationProps {
  id: string;
  message: string;
  type: 'info' | 'success' | 'warning' | 'error';
  onClose: () => void;
}

const ToastNotification: React.FC<ToastNotificationProps> = ({ 
  id, 
  message, 
  type,
  onClose 
}) => {
  return (
    <NotificationItem
      initial={{ opacity: 0, x: 50 }}
      animate={{ opacity: 1, x: 0 }}
      exit={{ opacity: 0, x: 50 }}
      transition={{ duration: 0.3 }}
      layout
    >
      <Alert
        severity={type}
        variant="filled"
        onClose={onClose}
        sx={{
          boxShadow: 3,
          width: '100%',
          '& .MuiAlert-message': {
            width: '100%',
          },
        }}
      >
        <AlertTitle sx={{ textTransform: 'capitalize' }}>{type}</AlertTitle>
        {message}
      </Alert>
    </NotificationItem>
  );
};

const NotificationSystem: React.FC = () => {
  const dispatch = useDispatch();
  const { notifications } = useSelector((state: RootState) => state.ui);
  
  // Only show the most recent 5 unread notifications
  const activeNotifications = notifications
    .filter(n => !n.read)
    .sort((a, b) => b.timestamp - a.timestamp)
    .slice(0, 5);
  
  const handleCloseNotification = (id: string) => {
    dispatch(markNotificationAsRead(id));
  };
  
  return (
    <NotificationContainer>
      <AnimatePresence>
        {activeNotifications.map((notification) => (
          <ToastNotification
            key={notification.id}
            id={notification.id}
            message={notification.message}
            type={notification.type}
            onClose={() => handleCloseNotification(notification.id)}
          />
        ))}
      </AnimatePresence>
    </NotificationContainer>
  );
};

export default NotificationSystem;
