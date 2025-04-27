import React, { useState, useRef, useEffect } from 'react';
import { 
  Box, 
  TextField, 
  IconButton, 
  Typography, 
  Avatar, 
  Paper,
  CircularProgress,
  Tooltip,
  useTheme
} from '@mui/material';
import { styled } from '@mui/material/styles';
import SendIcon from '@mui/icons-material/Send';
import MicIcon from '@mui/icons-material/Mic';
import CodeIcon from '@mui/icons-material/Code';
import SmartToyIcon from '@mui/icons-material/SmartToy';
import PersonIcon from '@mui/icons-material/Person';
import MoreVertIcon from '@mui/icons-material/MoreVert';
import { useSelector, useDispatch } from 'react-redux';
import { RootState } from '../../../store';
import { addMessage, setTyping } from '../../../store/slices/chatSlice';
import { motion, AnimatePresence } from 'framer-motion';
import { v4 as uuidv4 } from 'uuid'; // Note: need to add this dependency

const ChatContainer = styled(Box)({
  display: 'flex',
  flexDirection: 'column',
  height: '100%',
  overflow: 'hidden',
});

const MessagesContainer = styled(Box)({
  flexGrow: 1,
  overflowY: 'auto',
  padding: '16px',
  display: 'flex',
  flexDirection: 'column',
});

const MessageBubble = styled(Paper, {
  shouldForwardProp: (prop) => 
    prop !== 'isUser' && prop !== 'isTyping' && prop !== 'hasCode'
})<{ isUser: boolean; isTyping?: boolean; hasCode?: boolean }>(
  ({ theme, isUser, isTyping, hasCode }) => ({
    padding: '12px 16px',
    borderRadius: 16,
    maxWidth: '80%',
    marginBottom: 8,
    wordBreak: 'break-word',
    position: 'relative',
    ...(isUser
      ? {
          alignSelf: 'flex-end',
          backgroundColor: theme.palette.primary.main,
          color: theme.palette.primary.contrastText,
        }
      : {
          alignSelf: 'flex-start',
          backgroundColor:
            theme.palette.mode === 'dark'
              ? 'rgba(77, 208, 225, 0.15)'
              : 'rgba(77, 208, 225, 0.1)',
          border: `1px solid ${theme.palette.divider}`,
        }),
    ...(hasCode && {
      fontFamily: 'var(--font-code)',
    }),
    ...(isTyping && {
      '&::after': {
        content: '""',
        position: 'absolute',
        bottom: 8,
        right: isUser ? 8 : 'auto',
        left: isUser ? 'auto' : 8,
        width: 8,
        height: 8,
        borderRadius: '50%',
        backgroundColor: theme.palette.secondary.main,
        animation: 'pulse 1.5s infinite',
      },
      '@keyframes pulse': {
        '0%': {
          transform: 'scale(0.8)',
          opacity: 0.8,
        },
        '50%': {
          transform: 'scale(1.2)',
          opacity: 1,
        },
        '100%': {
          transform: 'scale(0.8)',
          opacity: 0.8,
        },
      },
    }),
  })
);

const MessageHeader = styled(Box)(({ theme }) => ({
  display: 'flex',
  alignItems: 'center',
  marginBottom: 8,
}));

const InputContainer = styled(Box)(({ theme }) => ({
  padding: '16px',
  borderTop: `1px solid ${theme.palette.divider}`,
  display: 'flex',
  alignItems: 'center',
}));

const MessageAvatar = styled(Avatar, {
  shouldForwardProp: (prop) => prop !== 'isUser'
})<{ isUser: boolean }>(({ theme, isUser }) => ({
  width: 32,
  height: 32,
  marginRight: isUser ? 0 : 8,
  marginLeft: isUser ? 8 : 0,
  backgroundColor: isUser
    ? theme.palette.secondary.main
    : theme.palette.primary.main,
}));

// Example messages for demo purposes
const exampleMessages = [
  {
    id: '1',
    content: 'Hello てゅん. How can I assist you today?',
    role: 'assistant',
    timestamp: Date.now() - 10000,
  },
  {
    id: '2',
    content: 'I need help optimizing this Python algorithm. It\'s running too slowly.',
    role: 'user',
    timestamp: Date.now() - 8000,
  },
  {
    id: '3',
    content: 'I\'d be happy to help optimize your algorithm. Could you share the code you\'re working with?',
    role: 'assistant',
    timestamp: Date.now() - 6000,
  },
  {
    id: '4',
    content: '```python\ndef fibonacci(n):\n    if n <= 1:\n        return n\n    return fibonacci(n-1) + fibonacci(n-2)\n\nresult = fibonacci(35)\nprint(result)\n```',
    role: 'user',
    timestamp: Date.now() - 4000,
  },
  {
    id: '5',
    content: 'I see the issue. You\'re using a recursive implementation of the Fibonacci sequence, which has exponential time complexity O(2^n). Let me optimize this for you with a more efficient approach using dynamic programming:',
    role: 'assistant',
    timestamp: Date.now() - 2000,
  },
  {
    id: '6',
    content: '```python\ndef fibonacci_optimized(n):\n    if n <= 1:\n        return n\n        \n    # Initialize array to store Fibonacci numbers\n    fib = [0] * (n + 1)\n    fib[0] = 0\n    fib[1] = 1\n    \n    # Calculate using bottom-up approach\n    for i in range(2, n + 1):\n        fib[i] = fib[i-1] + fib[i-2]\n        \n    return fib[n]\n\nresult = fibonacci_optimized(35)\nprint(result)\n```\n\nThis optimized version has a time complexity of O(n) and will run much faster for large values of n.',
    role: 'assistant',
    timestamp: Date.now() - 1000,
  }
];

const ChatInterface: React.FC = () => {
  const theme = useTheme();
  const dispatch = useDispatch();
  const { messages, isTyping } = useSelector((state: RootState) => state.chat);
  
  const [inputValue, setInputValue] = useState('');
  const messagesEndRef = useRef<HTMLDivElement>(null);
  
  // For demo purposes, use the example messages if no messages in state
  const displayMessages = messages.length > 0 ? messages : exampleMessages;
  
  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  };
  
  useEffect(() => {
    scrollToBottom();
  }, [displayMessages, isTyping]);
  
  const handleInputChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    setInputValue(e.target.value);
  };
  
  const handleSendMessage = () => {
    if (inputValue.trim()) {
      // Add user message
      dispatch(
        addMessage({
          id: uuidv4(),
          content: inputValue.trim(),
          role: 'user',
          timestamp: Date.now(),
        })
      );
      
      setInputValue('');
      
      // Simulate assistant typing
      dispatch(setTyping(true));
      
      // Simulate assistant response after a delay
      setTimeout(() => {
        dispatch(
          addMessage({
            id: uuidv4(),
            content: `I'm processing your request: "${inputValue.trim()}"`,
            role: 'assistant',
            timestamp: Date.now(),
          })
        );
        dispatch(setTyping(false));
      }, 2000);
    }
  };
  
  const handleKeyPress = (e: React.KeyboardEvent) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      handleSendMessage();
    }
  };
  
  const hasCodeContent = (content: string) => {
    return content.includes('```');
  };
  
  const formatMessageContent = (content: string) => {
    if (!content.includes('```')) {
      return content;
    }
    
    const parts = content.split('```');
    return (
      <>
        {parts.map((part, index) => {
          if (index % 2 === 0) {
            return <span key={index}>{part}</span>;
          } else {
            const languageMatch = part.match(/^(\w+)\n/);
            let code = part;
            
            if (languageMatch) {
              code = part.slice(languageMatch[0].length);
            }
            
            return (
              <Box
                key={index}
                sx={{
                  backgroundColor: 'rgba(0, 0, 0, 0.1)',
                  padding: 1.5,
                  borderRadius: 1,
                  fontFamily: 'var(--font-code)',
                  fontSize: '0.875rem',
                  overflowX: 'auto',
                  my: 1,
                }}
              >
                <Box sx={{ 
                  display: 'flex', 
                  justifyContent: 'space-between',
                  alignItems: 'center',
                  mb: 1,
                  opacity: 0.7,
                  fontSize: '0.75rem'
                }}>
                  <span>{languageMatch ? languageMatch[1] : 'code'}</span>
                  <span>code block</span>
                </Box>
                {code}
              </Box>
            );
          }
        })}
      </>
    );
  };
  
  return (
    <ChatContainer>
      <MessagesContainer>
        {/* Welcome Message */}
        <Box
          component={motion.div}
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.5 }}
          sx={{
            display: 'flex',
            flexDirection: 'column',
            alignItems: 'center',
            justifyContent: 'center',
            padding: 3,
            mb: 2,
          }}
        >
          <Avatar
            sx={{
              width: 64,
              height: 64,
              backgroundColor: 'primary.main',
              mb: 2,
            }}
          >
            <SmartToyIcon sx={{ fontSize: 36 }} />
          </Avatar>
          <Typography variant="h5" gutterBottom align="center">
            Jarviee AI Assistant
          </Typography>
          <Typography
            variant="body2"
            color="text.secondary"
            align="center"
            sx={{ maxWidth: 500 }}
          >
            I'm here to help with programming, knowledge exploration, and task management.
            How can I assist you today?
          </Typography>
        </Box>
        
        {/* Messages */}
        {displayMessages.map((message) => (
          <Box
            key={message.id}
            sx={{
              display: 'flex',
              flexDirection: message.role === 'user' ? 'row-reverse' : 'row',
              alignItems: 'flex-start',
              mb: 2,
            }}
            component={motion.div}
            initial={{ opacity: 0, y: 10 }}
            animate={{ opacity: 1, y: 0 }}
          >
            <MessageAvatar isUser={message.role === 'user'}>
              {message.role === 'user' ? <PersonIcon /> : <SmartToyIcon />}
            </MessageAvatar>
            <Box sx={{ maxWidth: 'calc(100% - 48px)' }}>
              <MessageBubble 
                isUser={message.role === 'user'} 
                hasCode={hasCodeContent(message.content)}
                elevation={0}
              >
                {formatMessageContent(message.content)}
              </MessageBubble>
              <Typography 
                variant="caption" 
                sx={{ 
                  ml: message.role === 'user' ? 0 : 1,
                  mr: message.role === 'user' ? 1 : 0,
                  textAlign: message.role === 'user' ? 'right' : 'left',
                  display: 'block',
                  color: 'text.secondary',
                  mt: 0.5,
                }}
              >
                {new Date(message.timestamp).toLocaleTimeString([], { 
                  hour: '2-digit', 
                  minute: '2-digit' 
                })}
              </Typography>
            </Box>
          </Box>
        ))}
        
        {/* Assistant typing indicator */}
        <AnimatePresence>
          {isTyping && (
            <Box
              sx={{
                display: 'flex',
                flexDirection: 'row',
                alignItems: 'flex-start',
                mb: 2,
              }}
              component={motion.div}
              initial={{ opacity: 0, y: 10 }}
              animate={{ opacity: 1, y: 0 }}
              exit={{ opacity: 0, y: 10 }}
            >
              <MessageAvatar isUser={false}>
                <SmartToyIcon />
              </MessageAvatar>
              <MessageBubble isUser={false} isTyping elevation={0}>
                <Box sx={{ display: 'flex', alignItems: 'center' }}>
                  <span>Thinking</span>
                  <Box
                    component={motion.div}
                    animate={{ y: [0, -5, 0] }}
                    transition={{
                      duration: 1.5,
                      repeat: Infinity,
                      repeatType: 'loop',
                      ease: 'easeInOut',
                      times: [0, 0.5, 1],
                      delay: 0,
                    }}
                    sx={{ mx: 0.5 }}
                  >
                    .
                  </Box>
                  <Box
                    component={motion.div}
                    animate={{ y: [0, -5, 0] }}
                    transition={{
                      duration: 1.5,
                      repeat: Infinity,
                      repeatType: 'loop',
                      ease: 'easeInOut',
                      times: [0, 0.5, 1],
                      delay: 0.2,
                    }}
                    sx={{ mx: 0.5 }}
                  >
                    .
                  </Box>
                  <Box
                    component={motion.div}
                    animate={{ y: [0, -5, 0] }}
                    transition={{
                      duration: 1.5,
                      repeat: Infinity,
                      repeatType: 'loop',
                      ease: 'easeInOut',
                      times: [0, 0.5, 1],
                      delay: 0.4,
                    }}
                    sx={{ mx: 0.5 }}
                  >
                    .
                  </Box>
                </Box>
              </MessageBubble>
            </Box>
          )}
        </AnimatePresence>
        
        <div ref={messagesEndRef} />
      </MessagesContainer>
      
      <InputContainer>
        <TextField
          fullWidth
          placeholder="Message Jarviee..."
          multiline
          maxRows={4}
          value={inputValue}
          onChange={handleInputChange}
          onKeyPress={handleKeyPress}
          variant="outlined"
          sx={{
            '& .MuiOutlinedInput-root': {
              borderRadius: 4,
              backgroundColor: theme.palette.mode === 'dark' 
                ? 'rgba(255, 255, 255, 0.05)' 
                : 'rgba(0, 0, 0, 0.03)',
            },
          }}
          InputProps={{
            endAdornment: (
              <Box sx={{ display: 'flex', alignItems: 'center' }}>
                <Tooltip title="Code snippet">
                  <IconButton size="small" color="primary">
                    <CodeIcon fontSize="small" />
                  </IconButton>
                </Tooltip>
                <Tooltip title="Voice input">
                  <IconButton size="small" color="primary">
                    <MicIcon fontSize="small" />
                  </IconButton>
                </Tooltip>
              </Box>
            ),
          }}
        />
        <IconButton
          color="primary"
          onClick={handleSendMessage}
          disabled={!inputValue.trim()}
          sx={{
            ml: 1,
            width: 48,
            height: 48,
            backgroundColor: 'primary.main',
            color: 'white',
            '&:hover': {
              backgroundColor: 'primary.dark',
            },
            '&.Mui-disabled': {
              backgroundColor: theme.palette.mode === 'dark' 
                ? 'rgba(255, 255, 255, 0.1)' 
                : 'rgba(0, 0, 0, 0.1)',
              color: theme.palette.mode === 'dark' 
                ? 'rgba(255, 255, 255, 0.3)' 
                : 'rgba(0, 0, 0, 0.3)',
            },
          }}
        >
          <SendIcon />
        </IconButton>
      </InputContainer>
    </ChatContainer>
  );
};

export default ChatInterface;
